import { execFile } from "node:child_process"
import { readFile, mkdir, access } from "node:fs/promises"
import path from "node:path"
import { promisify } from "node:util"
import { createHash } from "node:crypto"

const exec = promisify(execFile)
const same = (a, b) =>
  path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase()

// A refusal that has a known, user-readable cause. The prompt hook reports it into
// the session instead of failing the whole request with a generic HTTP 500.
class BlockedWorkspace extends Error {}

// Installed explicitly by the owner, outside mutable feature checkouts.
// No global auto-adoption: only repositories in the private local config.
export default {
  id: "local.agent-workspaces",
  async setup(ctx) {
    const options = ctx.options
    const config = JSON.parse(await readFile(options.config, "utf8"))
    const repo = config.repos.find((r) =>
      same(r.root, ctx.location.project.canonical))
    if (!repo) return
    const pending = new Map()

    // OpenCode's bundled strategy creates detached worktrees from the local
    // checkout. Own creation end-to-end, including calls from other sessions.
    await ctx.worktree.transform((editor) => {
      editor.add({
        id: "local.agent-workspaces.git",
        async create(input) {
          const root = path.join(repo.root, ".opencode", "worktrees")
          const directory = path.resolve(input.directory)
          if (!same(input.sourceDirectory, repo.root) || !same(path.dirname(directory), root))
            throw new Error("Workspace creation is limited to the configured repository and worktree root.")
          let fresh = true
          try {
            await exec("git", ["-C", repo.root, "fetch", "origin"], { windowsHide: true, timeout: 30_000 })
          } catch {
            // Keep offline work possible, but never fall back to the local branch:
            // it can lag behind the last fetched origin/main.
            fresh = false
            console.warn("Workspace: fetch failed; using the last fetched origin/main.")
          }
          // A bare "main" means the integration branch; the local one may lag behind.
          const ref = !input.branch || input.branch === "main" ? "origin/main" : input.branch
          if (ref.startsWith("-")) throw new Error("Invalid workspace starting ref.")
          const base = (await exec("git", ["-C", repo.root, "rev-parse", "--verify", `${ref}^{commit}`],
            { windowsHide: true, timeout: 10_000 })).stdout.trim()
          if (!/^[0-9a-f]{40,64}$/i.test(base)) throw new Error("Workspace base is not a commit.")
          const branch = `opencode/${path.basename(directory)}`
          const branchExists = await exec("git", ["-C", repo.root, "rev-parse", "--verify", "--quiet",
            `refs/heads/${branch}`], { windowsHide: true, timeout: 10_000 }).then(() => true, () => false)
          if (branchExists)
            throw new Error(`Workspace branch ${branch} already exists; no checkout was created.`)
          await exec("git", ["-C", repo.root, "worktree", "add", "-b", branch, directory, base],
            { windowsHide: true, timeout: 30_000 })
          const key = `owned-worktree:${directory.toLowerCase()}`
          await ctx.storage.set(key, { directory, fresh })
          if (!fresh) await ctx.storage.set(`workspace-stale:${directory.toLowerCase()}`, true)
          const head = (await exec("git", ["-C", directory, "rev-parse", "HEAD"],
            { windowsHide: true, timeout: 10_000 })).stdout.trim()
          if (head !== base) throw new Error("Workspace base mismatch; checkout was preserved.")
          return { directory }
        },
        async remove(input) {
          const root = path.join(repo.root, ".opencode", "worktrees")
          const directory = path.resolve(input.directory)
          if (!same(path.dirname(directory), root))
            throw new Error("Workspace removal is limited to the configured worktree root.")
          const key = `owned-worktree:${directory.toLowerCase()}`
          if (!await ctx.storage.get(key))
            throw new Error("Workspace ownership cannot be verified; checkout was preserved.")
          const args = ["-C", repo.root, "worktree", "remove"]
          if (input.force) args.push("--force")
          await exec("git", [...args, directory], { windowsHide: true, timeout: 30_000 })
          await ctx.storage.remove(key)
        },
        async list(sourceDirectory) {
          if (!same(sourceDirectory, repo.root)) return []
          const found = []
          let after
          do {
            const page = await ctx.storage.scan({ prefix: "owned-worktree:", after })
            for (const entry of page.entries) {
              const directory = entry.value?.directory
              if (typeof directory !== "string") continue
              try {
                await access(path.join(directory, ".git"))
                found.push({ directory, type: "worktree" })
              } catch { /* Keep ownership metadata; do not recreate or prune it. */ }
            }
            after = page.next || undefined
          } while (after)
          return found
        },
      })
    })

    async function maintenance(args) {
      await exec(options.python, ["-B", options.script, "--config", options.config, ...args], {
        windowsHide: true, timeout: 30_000,
      })
    }
    async function ensure(sessionID) {
      const result = await ctx.session.get({ sessionID })
      const session = result.data ?? result
      const directory = session.location.directory
      if (same(directory, repo.root)) {
        // Planning/read-only requests do not create checkouts. Build sessions use a
        // deterministic name, and the move control is admitted before tools run.
        if (session.agent === "plan" || session.agent === "explore") return
        const task = `task-${sessionID.replace(/[^a-zA-Z0-9-]/g, "-")}`
        const key = `workspace:${sessionID}`
        let created = await ctx.storage.get(key)
        if (!created) {
          created = await ctx.worktree.create({
            projectID: ctx.location.project.id, from: repo.root, name: task,
          })
          // Persist before requesting the move: a failed/delayed boundary must
          // never create another checkout on the next prompt.
          await ctx.storage.set(key, created)
        } else {
          await access(path.join(created.directory, ".git"))
        }
        const staleKey = `workspace-stale:${created.directory.toLowerCase()}`
        if (await ctx.storage.get(staleKey)) {
          await ctx.storage.remove(staleKey)
          try {
            await ctx.session.synthetic({
              sessionID,
              text: "Не удалось обновить origin/main по сети; рабочая копия создана от последней загруженной origin/main.",
            })
          } catch {
            console.warn("Workspace: could not deliver the stale-base warning to the session.")
          }
        }
        if (!created.prepared) {
          const status = await exec("git", ["-C", created.directory, "status", "--porcelain",
            "--untracked-files=all", "--ignored"], { windowsHide: true, timeout: 10_000 })
          if (status.stdout.trim()) throw new Error("New workspace is not clean; preserving it.")
          // The bundled Git strategy in V2.3.2 creates a detached checkout.
          // Give only this newly allocated clean checkout its own task branch.
          const branch = `opencode/${task}`
          const current = await exec("git", ["-C", created.directory, "branch", "--show-current"],
            { windowsHide: true, timeout: 10_000 })
          if (!current.stdout.trim()) {
            await exec("git", ["-C", created.directory, "switch", "-c", branch, "origin/main"],
              { windowsHide: true, timeout: 10_000 })
          } else if (current.stdout.trim() !== branch) {
            throw new Error("Unexpected workspace branch; preserving it.")
          }
          created = { ...created, prepared: true }
          await ctx.storage.set(key, created)
        }
        await ctx.session.move({ sessionID, directory: created.directory })
        // move() queues a control for the next safe boundary; it need not be
        // visible from inside this callback. The write guard below checks the
        // actual location before permitting any mutating tool.
        await maintenance(["register", "--tool", "opencode", "--session", sessionID,
          "--cwd", created.directory, "--project", ctx.location.project.id])
        return
      }
      const parent = path.dirname(directory)
      if (same(parent, path.join(repo.root, ".opencode", "worktrees"))) {
        // Registration needs a task branch. Name the real cause instead of the
        // generic "maintenance deferred", and never adopt an existing checkout.
        const current = await exec("git", ["-C", directory, "branch", "--show-current"],
          { windowsHide: true, timeout: 10_000 })
        if (!current.stdout.trim())
          throw new BlockedWorkspace(`Workspace ${path.basename(directory)} has a detached HEAD, so it cannot be ` +
            "registered and build tools stay blocked. Not a permission problem: from a session outside " +
            "this copy, create a task branch in it from origin/main (only if it is clean and holds " +
            "no unique commits), or make a new workspace.")
        await maintenance(["register", "--tool", "opencode", "--session", sessionID,
          "--cwd", directory, "--project", ctx.location.project.id])
      }
    }
    async function once(sessionID) {
      if (!pending.has(sessionID)) {
        const job = ensure(sessionID).finally(() => pending.delete(sessionID))
        pending.set(sessionID, job)
      }
      await pending.get(sessionID)
    }
    await ctx.session.hook("prompt", async (event) => {
      try {
        await once(event.sessionID)
      } catch (error) {
        if (!(error instanceof BlockedWorkspace)) throw error
        // Build tools stay blocked (tool.execute.before repeats this text); the
        // prompt itself goes through so the session can see and explain the cause.
        try {
          await ctx.session.synthetic({ sessionID: event.sessionID, text: error.message })
        } catch {
          console.warn(`Workspace: could not deliver the refusal to the session: ${error.message}`)
        }
      }
    })
    await ctx.command.transform((editor) => {
      editor.add({
        name: "workspace",
        description: "Prepare or resume this build session's worktree without a model call",
        execute: async ({ sessionID }) => { await once(sessionID) },
      })
      editor.add({
        name: "workspace-done",
        description: "Archive this finished task; safe housekeeping verifies its checkout separately",
        execute: async ({ sessionID }) => {
          const result = await ctx.session.get({ sessionID })
          const session = result.data ?? result
          if (!same(path.dirname(session.location.directory),
            path.join(repo.root, ".opencode", "worktrees"))) return
          await maintenance(["release", "--tool", "opencode", "--session", sessionID])
          await ctx.session.move({ sessionID, directory: repo.root })
          await ctx.session.update({ sessionID, archived: true })
        },
      })
    })
    // Reject tool execution until a move has actually taken effect. Never guess
    // that a queued move already isolated the main checkout.
    await ctx.tool.hook("execute.before", async (event) => {
      if (["read", "glob", "grep"].includes(event.tool)) return
      await once(event.sessionID)
      const result = await ctx.session.get({ sessionID: event.sessionID })
      const session = result.data ?? result
      if (same(session.location.directory, repo.root))
        throw new Error("Build tools require the task worktree; main checkout remains protected.")
    })
    await ctx.shell.hook("create.before", async (event) => {
      const worktrees = path.join(repo.root, ".opencode", "worktrees")
      const claudeWorktrees = path.join(repo.root, ".claude", "worktrees")
      const inside = (base, target) => {
        const rel = path.relative(base, path.resolve(target))
        return !rel || (!rel.startsWith("..") && !path.isAbsolute(rel))
      }
      // Session shell requests bypass tool.execute.before. Never allow them
      // to start in the integration checkout, including nested directories.
      if (inside(repo.root, event.cwd) && !inside(worktrees, event.cwd) &&
          !inside(claudeWorktrees, event.cwd))
        throw new Error("Shell requires a task worktree; main checkout remains protected.")
      const relative = path.relative(worktrees, path.resolve(event.cwd))
      if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) return
      const taskDirectory = path.join(worktrees, relative.split(path.sep)[0])
      // Test state lives outside the program and outside real user data.
      const key = createHash("sha256").update(
        path.resolve(taskDirectory).toLowerCase().replaceAll("\\", "/")).digest("hex").slice(0, 12)
      const sandbox = path.join(config.state, "sandboxes", key)
      await mkdir(path.join(sandbox, "data"), { recursive: true })
      await mkdir(path.join(sandbox, "temp"), { recursive: true })
      event.env.RUS_TRANSCRIBER_DATA = path.join(sandbox, "data")
      event.env.TEMP = path.join(sandbox, "temp")
      event.env.TMP = path.join(sandbox, "temp")
      event.env.PYTHONDONTWRITEBYTECODE = "1"
      event.env.PYTHONUTF8 = "1"
      event.env.PYTHONIOENCODING = "utf-8"
    })
    return () => {}
  },
}
