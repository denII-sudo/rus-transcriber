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
let archived = false
const ctx = {
  storage: { get: async (k) => storage.get(k), set: async (k, v) => storage.set(k, v) },
  options: { config: configFile, python, script },
  location: { project: { id: "synthetic", canonical: repo } },
  session: {
    get: async () => ({ agent, location: { directory } }),
    move: async (x) => {
      if (delayed) pendingDirectory = x.directory
      else directory = x.directory
    },
    update: async (x) => { archived = x.archived },
    hook: async (name, fn) => { hooks[name] = fn },
  },
  worktree: {
    create: async (x) => {
      creates++
      const out = path.join(repo, ".opencode", "worktrees", x.name)
      execFileSync("git", ["-C", repo, "worktree", "add", "--detach", out, "origin/main"])
      return { directory: out }
    },
  },
  command: { transform: async (fn) => fn({ add: (cmd) => { hooks[cmd.name] = cmd.execute } }) },
  tool: { hook: async (name, fn) => { hooks["tool-" + name] = fn } },
  shell: { hook: async (name, fn) => { hooks["shell-" + name] = fn } },
}
const dispose = await plugin.setup(ctx)
try {
  await hooks.workspace({ sessionID: "ses_synthetic" })
  const first = directory
  await hooks.prompt({ sessionID: "ses_synthetic" })
  assert.equal(directory, first)
  assert.equal(creates, 1)
  await hooks["tool-execute.before"]({ sessionID: "ses_synthetic", tool: "patch" })
  const shell = { cwd: directory, env: {} }
  await hooks["shell-create.before"](shell)
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
