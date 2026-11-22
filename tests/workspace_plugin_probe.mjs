// A real Node import with a fake OpenCode context. No server/model requests.
import assert from "node:assert/strict"
import { execFileSync } from "node:child_process"
import { readFile } from "node:fs/promises"
import path from "node:path"
import { pathToFileURL } from "node:url"

const [module, configFile, python, script] = process.argv.slice(2)
const plugin = (await import(pathToFileURL(module).href)).default
const config = JSON.parse(await readFile(configFile, "utf8"))
const repo = config.repos[0].root
const hooks = {}
const storage = new Map()
let directory = repo, creates = 0, agent = "build"
let delayed = false, pendingDirectory
let archived = false, warnings = 0, strategy
const ctx = {
  storage: { get: async (k) => storage.get(k), set: async (k, v) => storage.set(k, v),
    remove: async (k) => storage.delete(k),
    scan: async ({ prefix }) => ({ entries: [...storage].filter(([k]) => k.startsWith(prefix))
      .map(([key, value]) => ({ key, value })), next: null }) },
  options: { config: configFile, python, script },
  location: { project: { id: "synthetic", canonical: repo } },
  session: {
    get: async () => ({ agent, location: { directory } }),
    move: async (x) => {
      if (delayed) pendingDirectory = x.directory
      else directory = x.directory
    },
    update: async (x) => { archived = x.archived },
    synthetic: async () => { warnings++ },
    hook: async (name, fn) => { hooks[name] = fn },
  },
  worktree: {
    transform: async (fn) => fn({ add: (s) => { strategy = s } }),
    create: async (x) => {
      creates++
      const out = path.join(repo, ".opencode", "worktrees", x.name)
      return strategy.create({ sourceDirectory: repo, directory: out, branch: x.branch })
    },
  },
  command: { transform: async (fn) => fn({ add: (cmd) => { hooks[cmd.name] = cmd.execute } }) },
  tool: { hook: async (name, fn) => { hooks["tool-" + name] = fn } },
  shell: { hook: async (name, fn) => { hooks["shell-" + name] = fn } },
}
const dispose = await plugin.setup(ctx)
try {
  // Native worktree.create must work even when the main checkout is behind.
  const native = path.join(repo, ".opencode", "worktrees", "native-child")
  const base = execFileSync("git", ["-C", repo, "rev-parse", "origin/main"], { encoding: "utf8" }).trim()
  await strategy.create({ sourceDirectory: repo, directory: native })
  assert.equal(execFileSync("git", ["-C", native, "rev-parse", "HEAD"], { encoding: "utf8" }).trim(), base)
  assert.equal(execFileSync("git", ["-C", native, "branch", "--show-current"], { encoding: "utf8" }).trim(),
    "opencode/native-child")
  assert.equal((await strategy.list(repo))[0].directory, native)
  directory = native
  await hooks["tool-execute.before"]({ sessionID: "ses_native", tool: "shell" })
  await strategy.remove({ directory: native, force: false })
  assert.deepEqual(await strategy.list(repo), [])
  // An explicit bare "main" must not select the possibly lagging local branch.
  const explicit = path.join(repo, ".opencode", "worktrees", "explicit-main")
  await strategy.create({ sourceDirectory: repo, directory: explicit, branch: "main" })
  assert.equal(execFileSync("git", ["-C", explicit, "rev-parse", "HEAD"], { encoding: "utf8" }).trim(), base)
  await strategy.remove({ directory: explicit, force: false })
  // A pre-existing detached copy is not adopted; the refusal names the cause.
  const detached = path.join(repo, ".opencode", "worktrees", "detached-child")
  execFileSync("git", ["-C", repo, "worktree", "add", "--detach", detached, base])
  directory = detached
  await assert.rejects(hooks["tool-execute.before"]({ sessionID: "ses_detached", tool: "shell" }),
    /detached HEAD/)
  assert.equal(execFileSync("git", ["-C", detached, "branch", "--show-current"], { encoding: "utf8" }).trim(), "")
  execFileSync("git", ["-C", repo, "worktree", "remove", detached])
  execFileSync("git", ["-C", repo, "remote", "set-url", "origin", path.join(repo, "missing-remote")])
  const offline = path.join(repo, ".opencode", "worktrees", "offline-child")
  await strategy.create({ sourceDirectory: repo, directory: offline })
  assert.equal(execFileSync("git", ["-C", offline, "rev-parse", "HEAD"], { encoding: "utf8" }).trim(), base)
  await strategy.remove({ directory: offline, force: false })
  directory = repo
  // Offline ensure() must still isolate the session and warn it, using the
  // last fetched origin/main rather than the possibly older local branch.
  await hooks.workspace({ sessionID: "ses_synthetic" })
  assert.equal(warnings, 1)
  execFileSync("git", ["-C", repo, "remote", "set-url", "origin", path.join(path.dirname(repo), "remote.git")])
  const first = directory
  await hooks.prompt({ sessionID: "ses_synthetic" })
  assert.equal(directory, first)
  assert.equal(creates, 1)
  await hooks["tool-execute.before"]({ sessionID: "ses_synthetic", tool: "patch" })
  const shell = { cwd: directory, env: {} }
  await hooks["shell-create.before"](shell)
  await assert.rejects(hooks["shell-create.before"]({ cwd: repo, env: {} }))
  await assert.rejects(hooks["shell-create.before"]({ cwd: path.join(repo, "tests"), env: {} }))
  assert.ok(shell.env.RUS_TRANSCRIBER_DATA.startsWith(config.state))
  assert.equal(shell.env.TMP, shell.env.TEMP)
  const nested = { cwd: path.join(directory, "tests"), env: {} }
  await hooks["shell-create.before"](nested)
  assert.equal(nested.env.TEMP, shell.env.TEMP)
  directory = repo
  delayed = true
  await hooks.workspace({ sessionID: "ses_synthetic" })
  assert.equal(creates, 1) // A pending move reuses the persistent binding.
  await assert.rejects(hooks["tool-execute.before"]({ sessionID: "ses_synthetic", tool: "patch" }))
  directory = pendingDirectory
  delayed = false
  await hooks["tool-execute.before"]({ sessionID: "ses_synthetic", tool: "patch" })
  directory = repo
  agent = "plan"
  await hooks.prompt({ sessionID: "ses_plan" })
  await assert.rejects(hooks["tool-execute.before"]({ sessionID: "ses_plan", tool: "patch" }))
  await assert.rejects(hooks["tool-execute.before"]({ sessionID: "ses_plan", tool: "functions_shell" }))
  await assert.rejects(hooks["tool-execute.before"]({ sessionID: "ses_plan", tool: "unknown_mutator" }))
  await hooks["tool-execute.before"]({ sessionID: "ses_plan", tool: "read" })
  assert.equal(creates, 1)
  directory = first
  agent = "build"
  await hooks["workspace-done"]({ sessionID: "ses_synthetic" })
  assert.equal(directory, repo)
  assert.equal(archived, true)
  console.log("plugin: create/resume, write guard, sandbox and read-only plan passed")
} finally {
  dispose()
}
