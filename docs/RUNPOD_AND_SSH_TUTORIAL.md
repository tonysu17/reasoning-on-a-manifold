# The Terminal, SSH, and RunPod: A Ground-Up Tutorial

> A practical, self-contained introduction to the command line, SSH, and renting cloud GPUs
> on RunPod — written for an ML researcher who is comfortable in Python but wants to become
> genuinely fluent at the terminal and at driving remote machines. Everything is grounded in a
> concrete goal: getting from *your laptop* to *a GPU in the cloud running your inference*, and
> back again with your results.

*Written for macOS (your laptop runs zsh) connecting to Linux GPU pods. Conventions and most
commands are identical on Linux; Windows differences are noted where they matter. RunPod's web
UI changes over time — the **concepts** here are stable, but always copy the exact connect
command from the pod's own "Connect" dialog rather than memorising flags.*

---

## How to read this

The document has three movements, each building on the last:

1. **The terminal** (§1–§6) — what a shell actually is, navigating the filesystem, manipulating
   files, pipes and redirection, processes, and the two tools (`tmux`, package basics) that make
   remote work bearable. If you already live in the terminal, skim to §4.
2. **SSH** (§7–§11) — what SSH is, the public-key cryptography that underpins it, generating your
   own keys on your laptop, the `~/.ssh/config` file, moving files (`scp`/`rsync`), and port
   forwarding (the trick that lets you open a remote Jupyter in your local browser).
3. **RunPod** (§12–§18) — the mental model (Pods, templates, disks, billing), deploying a GPU,
   connecting to it the SSH way you just learned, a full end-to-end inference run, getting data
   in and out, and a cost-hygiene checklist so you never wake up to a surprise bill.

A recurring theme: **the cloud machine is just another Unix box you reach over an encrypted
pipe.** Once SSH clicks, "the cloud" stops being mysterious — it's the same `ls`, `cd`, and
`python` you already run, only the computer is somewhere else.

---

## 1. What the terminal actually is

When people say "the terminal" they usually conflate three distinct things. Pulling them apart
removes a lot of confusion.

- **The terminal emulator** is the *application* with a window — on your Mac that's **Terminal.app**
  or **iTerm2** (or the integrated terminal inside VS Code). Historically a "terminal" was a
  physical device (a screen + keyboard) wired to a mainframe; today the app *emulates* that
  device. Its only job is to draw text, capture your keystrokes, and pass them along.
- **The shell** is the *program running inside* that window that actually interprets what you
  type. On modern macOS the default shell is **zsh** (the "Z shell"); older Macs and most Linux
  servers default to **bash**. The shell is a full programming language: it reads a line, expands
  it (globs, variables), finds the right program, runs it, and shows you the output. When this
  document says "run X," it means "type X at the shell prompt and press Enter."
- **The programs** you run — `ls`, `python`, `git`, `ssh` — are separate executables the shell
  launches on your behalf. The shell is the receptionist; the programs do the work.

The **prompt** is the bit of text the shell prints when it's ready for input, e.g.
`tonysu@MacBook ~ %`. The trailing `%` is zsh's default; bash uses `$`. Throughout this doc I'll
write a leading `$` to mean "type this at your prompt" — **don't type the `$` itself.**

You can check which shell you're in:

```bash
echo $SHELL          # prints e.g. /bin/zsh — your login shell
```

`echo` prints its arguments; `$SHELL` is an *environment variable* (more in §5) holding the path
to your shell. This single line already shows the shell's three core moves: it expanded `$SHELL`
into a value, ran the `echo` program, and printed the result.

---

## 2. Navigating the filesystem

A Unix filesystem is a single tree rooted at `/` (the "root"). There are no drive letters like
Windows's `C:`; everything hangs off `/`. Your personal files live under your **home directory**,
which on macOS is `/Users/tonysu` and is abbreviated `~`.

### Absolute vs relative paths

- An **absolute path** starts at the root and is unambiguous from anywhere: `/Users/tonysu/Documents`.
- A **relative path** is interpreted *from wherever you currently are*: `Documents/notes.txt`
  means "the `Documents` folder inside my current directory, then `notes.txt`."

Four special names you'll use constantly:

| Symbol | Means |
|---|---|
| `~` | your home directory (`/Users/tonysu`) |
| `.` | the current directory |
| `..` | the parent directory (one level up) |
| `/` | the root, or a separator between path components |

### The core navigation commands

```bash
pwd                       # "print working directory" — where am I right now?
ls                        # list files in the current directory
ls -l                     # long format: permissions, owner, size, modified date
ls -a                     # include hidden files (those whose name starts with a dot)
ls -lah                   # combine flags; 'h' = human-readable sizes (4.0K, 2.3G)
cd Documents              # change directory into Documents (relative)
cd /Users/tonysu/Desktop  # change directory (absolute)
cd ..                     # go up one level
cd ~                      # go home (cd with no argument also goes home)
cd -                      # go back to the previous directory you were in
```

**Flags** (also called options) are the `-l`, `-a` bits — they modify a command's behaviour.
Single-letter flags can usually be combined: `ls -l -a -h` ≡ `ls -lah`. Many commands also have
long-form flags like `ls --all`. When unsure what flags exist, ask the program:

```bash
ls --help                 # quick summary (works for most GNU tools; sparse on macOS's BSD ls)
man ls                    # the full manual page — press 'q' to quit, '/' to search inside it
```

`man` (manual) is your most important reference. Reading `man ssh` and `man rsync` later in this
doc will teach you more than any blog post.

### Tab completion — the habit that changes everything

Press **Tab** while typing a path or command and the shell completes it for you. Type `cd Doc`
then Tab → it fills in `Documents/`. If there are multiple matches, press Tab twice to list them.
This is not a minor convenience: it eliminates typos in long paths, and it's the difference
between the terminal feeling clumsy and feeling fast. Use it relentlessly.

---

## 3. Creating, viewing, moving, and deleting files

```bash
mkdir results                 # make a directory
mkdir -p data/raw/2026        # -p makes parents as needed (data, then raw, then 2026)
touch notes.txt               # create an empty file (or update its timestamp if it exists)

cat notes.txt                 # dump a file's whole contents to the screen
less notes.txt                # page through a long file: arrows/space to scroll, 'q' to quit
head notes.txt                # first 10 lines
head -n 50 notes.txt          # first 50 lines
tail notes.txt                # last 10 lines
tail -f train.log             # -f = "follow": keep printing new lines as they're appended
                              #   (perfect for watching a running job's log live)

cp source.txt dest.txt        # copy a file
cp -r olddir newdir           # -r = recursive: copy a directory and everything in it
mv old.txt new.txt            # move OR rename (same operation in Unix)
mv file.txt ~/Documents/      # move a file into a directory

rm file.txt                   # remove a file — THERE IS NO TRASH/UNDO
rm -r somedir                 # remove a directory and its contents, recursively
rm -rf somedir                # -f = force, no prompts. Powerful and dangerous.
```

> **The `rm -rf` warning, stated once and seriously.** `rm` does not move things to a Trash —
> it deletes immediately and irreversibly. `rm -rf /` or `rm -rf ~` can erase your machine.
> Before running any `rm -rf`, read the path twice, and prefer to `ls` the target first to see
> exactly what you're about to destroy. On a cloud pod the danger is smaller (it's disposable),
> but the habit is worth building on your laptop.

### Editing text at the terminal

You'll eventually need to edit a config file on a remote machine that has no GUI. Two editors are
on essentially every Linux box:

- **`nano`** — beginner-friendly. `nano config.yaml` opens it; the bottom bar shows commands
  (`^O` means Ctrl-O to save/"WriteOut", `^X` to exit). Start here.
- **`vim`** — ubiquitous and powerful but modal, with a real learning curve. The survival kit:
  `vim file` to open; press `i` to enter *insert* mode and type normally; press `Esc` to return
  to *normal* mode; type `:wq` then Enter to write-and-quit, or `:q!` to quit without saving.
  Knowing just those four things means you're never trapped in vim (a famous beginner panic).

---

## 4. Pipes, redirection, and composing tools

The Unix philosophy is "small tools that do one thing, combined." The glue is three ideas. To use
them you need one concept: every program has three default streams — **stdin** (input, stream 0),
**stdout** (normal output, stream 1), and **stderr** (error output, stream 2).

### Redirection — sending streams to/from files

```bash
python run.py > out.log       # redirect stdout INTO out.log (overwrites the file)
python run.py >> out.log      # append to out.log instead of overwriting
python run.py 2> err.log      # redirect stderr (stream 2) into err.log
python run.py > out.log 2>&1  # stdout to out.log, and stderr to "the same place as stdout"
python run.py &> all.log      # shorthand: both stdout and stderr into one file
sort < names.txt              # feed names.txt into sort's stdin
```

`> out.log 2>&1` is an idiom worth memorising — it captures *everything* a program prints into
one log file, which is exactly what you want when launching a long run you'll inspect later.

### Pipes — wiring one program's output into the next's input

The `|` ("pipe") character connects stdout of the left command to stdin of the right command, with
no temporary file:

```bash
ls -l | grep ".py"            # list files, keep only lines containing ".py"
cat train.log | grep -i error # show only lines mentioning "error" (-i = case-insensitive)
history | grep ssh            # find past commands you ran that mention ssh
ps aux | grep python          # find running python processes (see §6)
du -h --max-depth=1 | sort -h # sizes of subdirectories, sorted smallest→largest
```

A few tools you'll pipe into constantly:

- **`grep`** — print lines matching a pattern. `grep -i` ignores case; `grep -r` searches
  recursively through a directory; `grep -n` shows line numbers; `grep -v` *inverts* (lines that
  *don't* match).
- **`wc`** — count things. `wc -l` counts lines (e.g. `ls | wc -l` = "how many files here?").
- **`sort`** and **`uniq`** — order lines / collapse duplicates.
- **`find`** — locate files by name/size/age: `find . -name "*.npy"` finds every `.npy` file in
  the current tree.

The skill to build is *thinking in pipelines*: "list the files → filter to the ones I care about
→ count them" becomes `ls | grep pattern | wc -l`. Each stage is dumb; the composition is powerful.

---

## 5. Environment variables and the `PATH`

An **environment variable** is a named value the shell and the programs it launches can read. You
saw `$SHELL` already. The most important one is **`PATH`**.

```bash
echo $PATH                    # a colon-separated list of directories
```

When you type `python`, the shell does not search your whole disk — it walks the directories in
`PATH`, left to right, and runs the *first* `python` it finds. This is why "command not found"
usually means "the program isn't installed, or its folder isn't on your PATH," and why installing
tools sometimes requires adding a directory to PATH.

```bash
which python                  # show WHICH python the shell would actually run
which -a python               # show all of them on the PATH, in priority order
export MY_VAR=hello           # set a variable for this shell and its child programs
echo $MY_VAR                  # → hello
export PATH="$HOME/bin:$PATH" # prepend a directory to PATH (note: keep the old $PATH!)
```

`export` makes a variable available to programs the shell launches (child processes); without
`export` it's only visible to the shell itself. Two more you'll meet:

- **`$HOME`** — your home directory (same as `~`).
- **`CUDA_VISIBLE_DEVICES`** — set this to control which GPUs a process can see, e.g.
  `CUDA_VISIBLE_DEVICES=0 python infer.py` restricts the run to GPU 0. Very useful on multi-GPU pods.

### Where settings persist: your shell config file

Variables set with `export` vanish when you close the terminal. To make them permanent, put them
in your shell's startup file, which zsh reads every time it starts: **`~/.zshrc`** (bash uses
`~/.bashrc` or `~/.bash_profile`). Add a line like `export PATH="$HOME/bin:$PATH"`, save, then
either reopen the terminal or run `source ~/.zshrc` to re-read it immediately. The same idea
applies on a remote pod — though because pods are often ephemeral, you may instead put setup in a
script you run on each new pod (see §16).

---

## 6. Processes, jobs, and keeping work alive

A running program is a **process** with a numeric **PID** (process ID). Managing them is essential
once you launch long jobs.

```bash
ps aux                        # list ALL processes (a is all users, u is detailed, x includes daemons)
ps aux | grep python          # filter to your python processes
top                           # live, auto-updating table of processes by CPU/memory; 'q' to quit
htop                          # nicer colourful version if installed (often need to install it)
kill 12345                    # politely ask process with PID 12345 to terminate (sends SIGTERM)
kill -9 12345                 # forcefully kill it (SIGKILL) — last resort, no cleanup
```

### Foreground, background, and signals

When you run `python long_job.py`, it occupies your terminal (the **foreground**) — you can't type
other commands and the keyboard goes to that program.

- **Ctrl-C** sends an *interrupt* (SIGINT) to the foreground program, asking it to stop. This is
  how you abort a run.
- **Ctrl-Z** *suspends* it (pauses, doesn't kill). Then `bg` resumes it in the **background** so
  you get your prompt back; `fg` brings it back to the foreground; `jobs` lists suspended/background
  jobs.
- Appending **`&`** launches a command in the background from the start: `python job.py &`.

```bash
python job.py > job.log 2>&1 &   # run in background, all output to a log, prompt returns immediately
jobs                             # see background jobs in this shell
```

### The disconnection problem — and `tmux` (read this twice)

Here is the trap that bites everyone new to cloud machines. When you SSH into a pod and start a
job, that job is a *child of your SSH session*. **If your SSH connection drops** — laptop sleeps,
Wi-Fi blips, you close the lid — **the session dies and typically takes your job with it.** Hours
of inference, gone.

The fix is a **terminal multiplexer**: a program that runs *on the remote machine* and hosts your
shell sessions independently of your connection. You attach to it over SSH; if you detach (or get
disconnected), the sessions keep running on the server. Reconnect later and reattach exactly where
you were. The standard tool is **`tmux`** (its older cousin is `screen`).

```bash
tmux                          # start a new tmux session (you're now "inside" it)
# ... launch your long job normally, e.g. python infer.py ...
# Detach: press Ctrl-b, release, then press d.  Your job keeps running.

tmux ls                       # list running tmux sessions
tmux attach                   # reattach to the last session
tmux attach -t 0              # attach to a specific session by name/number
tmux new -s infer             # start a named session ("infer")
tmux attach -t infer          # reattach to it by name
```

The key combo `Ctrl-b` is tmux's "prefix" — you press it, let go, then press a command key.
`Ctrl-b d` detaches; `Ctrl-b c` creates a new window; `Ctrl-b "` splits the pane horizontally;
`Ctrl-b %` splits vertically; `Ctrl-b` then arrow keys move between panes. You don't need more than
*new / detach / attach* to get the safety benefit.

**Rule of thumb for cloud GPUs: the very first thing you do after connecting is start `tmux`,
then do all your work inside it.** This single habit prevents the most common and most painful
beginner disaster. (Note: tmux protects against *connection* loss — it does **not** protect against
the pod itself being stopped or terminated; that's a billing/lifecycle matter covered in §17.)

---

## 7. What SSH actually is

**SSH** stands for **Secure Shell**. It is a network *protocol* (and the `ssh` program that speaks
it) for securely operating a remote computer's command line over an untrusted network like the
internet. When you SSH into a RunPod GPU, you get a shell prompt that behaves exactly like the
local one in §1 — `ls`, `cd`, `python` — except the computer executing those commands is in a
datacenter, possibly on another continent. Your keystrokes travel to it encrypted, and its output
travels back encrypted.

It replaced an ancient, insecure tool called `telnet` that sent everything — including passwords —
as plain readable text. SSH provides three guarantees over the wire:

1. **Confidentiality** — all traffic is encrypted, so a network eavesdropper sees only ciphertext.
2. **Integrity** — tampering with the data in transit is detected.
3. **Authentication** — *both* sides prove who they are: you prove your identity to the server,
   and (crucially, and often overlooked) the server proves its identity to you, so you can't be
   tricked into connecting to an impostor.

SSH conventionally listens on **TCP port 22** (we'll explain ports in §9). The program lives at
`/usr/bin/ssh` on your Mac and is already installed — nothing to set up to *use* it.

The basic shape of the command is:

```bash
ssh username@hostname
```

`username` is the account on the *remote* machine (on RunPod pods this is typically `root`);
`hostname` is the remote machine's address (an IP like `213.181.x.x` or a name like
`ssh.runpod.io`). You'll often add `-p PORT` (non-standard port) and `-i KEYFILE` (which key to
use) — both of which the next sections explain.

---

## 8. The heart of SSH: public-key cryptography

You *can* log in to an SSH server with a password, but the professional and RunPod-standard way is
with a **key pair**. Understanding this is the conceptual core of the whole tutorial, so we'll go
slowly. It's worth it: once this clicks, RunPod's connection instructions become obvious instead
of magic.

### The idea: two keys, one stays secret

Public-key (a.k.a. *asymmetric*) cryptography uses a mathematically linked **pair** of keys:

- A **private key** — a file that lives **only on your laptop** and that you **never share with
  anyone, ever.** Treat it like the physical key to your house.
- A **public key** — a short string you can hand out freely, paste into web forms, and copy onto
  any server. Think of it as a *padlock* that only your private key can open.

The magic property: anything "locked" using the public key can only be "unlocked" by the
corresponding private key, and possession of the public key gives an attacker no feasible way to
derive the private key. So you can safely scatter your public key across the world.

### How a login actually works (the challenge–response)

When you connect, authentication proceeds roughly like this:

1. You've previously placed your **public** key on the server, in a file the server keeps called
   `~/.ssh/authorized_keys` (RunPod does this injection for you — §13).
2. On connection, the server says, in effect: "Prove you hold the private key matching one of the
   public keys I trust. Here is a random challenge."
3. Your `ssh` client uses your **private** key to compute a signature over that challenge and sends
   it back. The private key itself **never leaves your laptop** and never crosses the network.
4. The server verifies the signature using your **public** key. If it checks out, you're in.

This is strictly safer than a password: nothing reusable is ever transmitted, so even a
compromised network can't capture a credential to replay. It's also more convenient — no password
to type each time (and you can add a passphrase that's cached locally, see §10).

### Host keys: how *you* verify the *server*

Authentication runs both ways. The *server* also has its own key pair (its **host key**). The
first time you connect, `ssh` shows you the server's host-key **fingerprint** and asks:

```
The authenticity of host '...' can't be established.
ED25519 key fingerprint is SHA256:abcd1234...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

When you type `yes`, that host key is recorded in `~/.ssh/known_hosts`. On every future connection,
`ssh` checks the server still presents the same host key. If it ever *changes*, ssh loudly refuses
to connect (`REMOTE HOST IDENTIFICATION HAS CHANGED!`) — protecting you from a man-in-the-middle
impostor. For ephemeral cloud pods this warning sometimes appears legitimately because a *new* pod
reuses an IP/port a previous pod used; the fix in that specific, expected case is to remove the
stale entry: `ssh-keygen -R '[host]:port'`. (Only do this when you understand *why* it changed —
on a long-lived server, an unexpected host-key change is a genuine red flag.)

---

## 9. Generating your SSH keys on your laptop

Now the hands-on part. You'll create a key pair once, on your Mac, and reuse it for RunPod and any
other server forever.

### Step 1 — check whether you already have one

```bash
ls -al ~/.ssh
```

`~/.ssh` is the hidden directory where SSH keeps its files. Look for a pair like
`id_ed25519` (private) and `id_ed25519.pub` (public), or older `id_rsa` / `id_rsa.pub`. If they
exist, you can reuse them and skip to §10. If the directory doesn't exist or is empty, make a key.

### Step 2 — generate an Ed25519 key pair

```bash
ssh-keygen -t ed25519 -C "tonysu@macbook 2026"
```

- `-t ed25519` chooses the **Ed25519** algorithm — modern, fast, very secure, with short keys.
  Prefer it over RSA. (If some old system rejects Ed25519, fall back to `-t rsa -b 4096`.)
- `-C "..."` adds a **comment** (free text, conventionally an email or a note) that gets appended
  to the public key so you can tell your keys apart later. It has no security role.

It will prompt:

```
Enter file in which to save the key (/Users/tonysu/.ssh/id_ed25519):   ← press Enter to accept default
Enter passphrase (empty for no passphrase):                            ← see note below
Enter same passphrase again:
```

**The passphrase** encrypts the private key *file on disk*, so that if someone steals the file they
still can't use it without the passphrase. Strongly recommended. The minor inconvenience of typing
it is removed by the `ssh-agent` (§10), which caches it for your session. You *may* leave it empty
for pure convenience on a personal laptop, but a passphrase is the safer default.

This creates two files:

```bash
ls -l ~/.ssh
# id_ed25519       ← PRIVATE key. Never share. Never copy to a server. Never paste anywhere.
# id_ed25519.pub   ← PUBLIC key. Safe to share; this is what you give RunPod.
```

### Step 3 — fix permissions (SSH is strict about this)

SSH refuses to use keys that are readable by other users on the machine — a safety feature. Ensure:

```bash
chmod 700 ~/.ssh                 # directory: only you can read/enter it
chmod 600 ~/.ssh/id_ed25519      # private key: only you can read/write it
chmod 644 ~/.ssh/id_ed25519.pub  # public key: world-readable is fine
```

(`chmod` sets file permissions; the digits are explained in §11. If ssh ever complains
"UNPROTECTED PRIVATE KEY FILE," rerun the `600` line.)

### Step 4 — view your public key (you'll paste this into RunPod)

```bash
cat ~/.ssh/id_ed25519.pub
```

It prints one line like:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILongBase64String... tonysu@macbook 2026
```

That entire line *is* your public key. Copy it whole (on macOS you can even do
`pbcopy < ~/.ssh/id_ed25519.pub` to put it straight on your clipboard). This is the string you'll
give to RunPod in §13. **Do not** ever paste the contents of the file *without* `.pub` — that's the
private key.

---

## 10. The SSH agent and the `~/.ssh/config` file (quality-of-life)

Two pieces of polish make daily SSH painless.

### `ssh-agent` — type your passphrase once

The **ssh-agent** is a small background program that holds your *decrypted* private key in memory
for the session, so you enter the passphrase once instead of on every connection. On macOS it
integrates with the Keychain:

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # add key; store passphrase in macOS Keychain
ssh-add -l                                        # list keys currently loaded in the agent
```

After this, connections "just work" without re-prompting. (On plain Linux it's `ssh-add
~/.ssh/id_ed25519`; the agent usually starts automatically with your desktop session.)

### `~/.ssh/config` — give servers friendly names

Typing `ssh -i ~/.ssh/id_ed25519 -p 41953 root@213.181.x.x` every time is miserable and
error-prone. The `~/.ssh/config` file lets you define an **alias** with all the settings baked in.
Create or edit it:

```bash
nano ~/.ssh/config
```

and add a block (the indentation is conventional, not required):

```ssh-config
Host runpod
    HostName 213.181.x.x
    User root
    Port 41953
    IdentityFile ~/.ssh/id_ed25519
    # keep the connection alive through brief network hiccups:
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

Now the entire connection is just:

```bash
ssh runpod
```

Everything else (`scp`, `rsync`, port forwards) can also refer to `runpod` instead of the long
form. Each time you deploy a *new* pod you update the `HostName`/`Port` lines (RunPod gives you new
ones per pod) and keep using the same short alias. Set the permissions once: `chmod 600 ~/.ssh/config`.

---

## 11. Reading file permissions (the `ls -l` columns)

You'll see permission strings constantly on Unix, and SSH cares about them, so decode them once.
`ls -l ~/.ssh` shows lines like:

```
-rw-------  1 tonysu staff  411 Jun 28 14:02 id_ed25519
-rw-r--r--  1 tonysu staff   99 Jun 28 14:02 id_ed25519.pub
drwx------  5 tonysu staff  160 Jun 28 14:02 .
```

The leading 10 characters are the type + permissions:

- **Char 1**: `-` = regular file, `d` = directory, `l` = symbolic link.
- **Chars 2–4**: permissions for the **owner** (you): `r`ead, `w`rite, e`x`ecute.
- **Chars 5–7**: permissions for the **group**.
- **Chars 8–10**: permissions for **others** (everyone else).

So `-rw-------` means "a file the owner can read and write, and nobody else can touch" — exactly
right for a private key. `-rw-r--r--` means "owner read/write, group and others read-only" — fine
for a public key.

`chmod` sets these, either symbolically (`chmod u+x script.sh` = add execute for the user) or with
the **octal** notation you saw in §9, where each digit is a sum of read=4, write=2, execute=1:

| Octal | Means | rwx |
|---|---|---|
| `7` | read+write+execute | `rwx` |
| `6` | read+write | `rw-` |
| `5` | read+execute | `r-x` |
| `4` | read only | `r--` |
| `0` | nothing | `---` |

`chmod 600 file` = owner `6` (rw), group `0`, others `0` → `-rw-------`. `chmod 700 dir` gives the
owner full access to a directory and nobody else. That's why §9 used `600` for the private key and
`700` for the `.ssh` directory.

---

## 12. RunPod: the mental model

RunPod is a marketplace that rents you GPU machines by the second. Before clicking anything, get
the vocabulary straight — most beginner confusion (and surprise charges) comes from mixing these up.

### Pods vs Serverless

- A **Pod** is a GPU **container** (a lightweight virtual machine) that you rent and keep running —
  you SSH in, install things, and treat it like your own Linux box. This is what you want for
  interactive research, development, and running your inference scripts. The rest of this tutorial
  is about Pods.
- **Serverless** is a different product: you package your code as an endpoint that RunPod spins up
  *on demand* per request and bills only while it runs. Great for deploying an inference *API* to
  others; overkill for your own exploratory runs. Ignore it for now.

### Secure Cloud vs Community Cloud

When deploying a Pod you choose a tier:

- **Secure Cloud** — GPUs in vetted, high-reliability datacenters. More expensive, more stable,
  better networking. Prefer it for anything you care about not being interrupted.
- **Community Cloud** — capacity from vetted third-party hosts. Cheaper, but variability in
  reliability and network speed. Fine for cheap, interruptible experiments.

### The two kinds of disk (this trips everyone up)

A Pod has **two** storage areas, and the difference is about *what survives*:

- **Container disk** — scratch space tied to the container's lifetime. It's wiped if the pod is
  reset/terminated. Don't keep anything precious here. Sized in GB when you deploy.
- **Volume disk** — a persistent volume mounted at **`/workspace`** that survives a *stop* and
  restart of the same pod. Put your code, data, and results here. (You still lose it if you
  *terminate* the pod — see §17.)
- **Network volume** *(optional)* — a separate, named volume you can detach from one pod and attach
  to another, so your data outlives any individual pod. The right choice if you'll spin pods up and
  down repeatedly and want a permanent home for your datasets and checkpoints. There's a small
  ongoing storage charge for it.

**Practical rule:** keep everything you'd be sad to lose under `/workspace` (or a network volume),
and treat the rest of the filesystem as disposable.

### Templates and images

A **template** preselects a Docker **image** (the OS + preinstalled software) and default ports.
For ML, choose an official **RunPod PyTorch** template — it ships a recent Ubuntu, CUDA, Python,
and PyTorch already working together, so you skip the painful CUDA/driver matching described in the
GPU primer. You can also bring your own image, but start with the official PyTorch one.

### Billing model (read before you deploy)

- You're billed **per second** while a pod is **running**, at the GPU's hourly rate ÷ 3600.
- **On-demand** pods run until you stop them. **Spot/interruptible** pods are cheaper but can be
  reclaimed with little notice if someone outbids you — fine for fault-tolerant or checkpointed work.
- **Stopped pods still cost money for storage** (the volume disk persists, and you pay a small rate
  to keep it). **Terminated** pods cost nothing but are gone forever. §17 makes this concrete.
- You prepay credits; when they run out, pods stop. (Recall your project note: keep an eye on
  credit balance so a long run isn't cut off mid-way.)

---

## 13. One-time RunPod setup: account, credits, and your SSH key

1. **Create an account** at runpod.io and **add credits** (billing → add funds). Nothing runs
   without a balance.
2. **Register your public SSH key with your account** so RunPod injects it into every pod you
   deploy. In the RunPod console go to **Settings → SSH Public Keys** and paste the *entire* line
   you printed in §9 with `cat ~/.ssh/id_ed25519.pub` (the `ssh-ed25519 AAAA... comment` string).
   - This is the public key — pasting it into a web form is exactly what public keys are for.
   - Doing this *before* you deploy means new pods automatically trust your laptop. (If you add the
     key *after* a pod is already running, you may need to redeploy or add the key into that pod's
     `~/.ssh/authorized_keys` manually.)
3. *(Optional but recommended)* Install the **`runpodctl`** CLI on your laptop for managing pods and
   moving files from the terminal (§16). RunPod provides an install command in its docs; once
   installed you authenticate it with an API key from **Settings → API Keys**.

---

## 14. Deploying your first GPU Pod

In the RunPod console:

1. Go to **Pods → Deploy** (or "Deploy a Pod").
2. **Choose a GPU.** For a 1.5B-class reasoning model doing inference, you do *not* need an H100 —
   a single mid-range card (e.g. an RTX 4090, L4, or A40) has ample VRAM and is far cheaper; step
   up to an A100/H100 only if you need the speed or larger models. (Use the memory math from §6 of
   the GPU primer to size it: weights + KV cache + overhead.) Each option shows its hourly price and
   VRAM.
3. **Choose Secure vs Community** cloud (§12) per your reliability/price preference.
4. **Pick the PyTorch template.**
5. **Set disk sizes.** Give the **container disk** enough for packages (e.g. 20–40 GB) and the
   **volume disk** (`/workspace`) enough for your code, model weights, and outputs (model weights
   for a 1.5B model are a few GB; size generously if you'll cache several).
6. *(Optional)* Attach a **network volume** if you want data to outlive this pod.
7. **Deploy.** The pod enters "provisioning," then "running." Billing starts when it's running.

Once running, open the pod's **Connect** dialog. RunPod shows you several ways in:

- A **web terminal** (a shell in your browser) — handy for a quick look, no setup.
- **JupyterLab** via an HTTPS proxy URL (often something like
  `https://<pod-id>-8888.proxy.runpod.net`) — a browser IDE/notebook.
- **SSH** — the two variants explained next. This is what you came here to learn.

---

## 15. Connecting to your Pod over SSH

The Connect dialog gives two SSH options. Know the difference.

### Option A — SSH over the exposed TCP port (the full-featured one)

If your pod exposes a public TCP port for SSH, the dialog shows a command like:

```bash
ssh root@213.181.x.x -p 41953 -i ~/.ssh/id_ed25519
```

Read it with your §7–§9 knowledge:

- `root` — the remote username on the pod.
- `213.181.x.x` — the pod's public IP address.
- `-p 41953` — connect to **port** 41953 rather than the default 22 (RunPod maps each pod's SSH to a
  unique high-numbered port on a shared IP). A **port** is just a numbered channel on a machine, so
  many services can share one IP address; SSH conventionally uses 22, but here it's remapped.
- `-i ~/.ssh/id_ed25519` — use *this* private key to authenticate (the `-i` = "identity file").

This variant supports the **full** SSH feature set — `scp`, `rsync`, and **port forwarding** — so
prefer it. **Copy the exact command from the dialog** (the IP and port are unique to your pod) and
paste it into your laptop terminal. On first connect you'll get the host-key prompt from §8 — type
`yes`. You're now at a shell *on the GPU*.

To fold it into your `~/.ssh/config` (§10) so you can just type `ssh runpod`, set `HostName` to the
IP, `Port` to the number, `User root`, and `IdentityFile ~/.ssh/id_ed25519`.

### Option B — SSH via the RunPod proxy (the basic one)

The dialog also shows a proxy form like:

```bash
ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

This routes through RunPod's gateway (`ssh.runpod.io`) and works even when the pod has no public
TCP port. It's perfectly fine for an interactive shell, but it's **limited** — notably it does not
support `scp`/`rsync` file transfer or arbitrary port forwarding. Use it as a fallback; use Option A
when you need to move files or tunnel.

### First things to run after connecting

```bash
nvidia-smi                    # confirm the GPU is present, see its model, VRAM, driver/CUDA version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # sanity-check PyTorch sees CUDA
df -h /workspace              # check free space on your persistent volume
tmux new -s work              # START TMUX before doing anything long (see §6!)
```

`nvidia-smi` ("NVIDIA System Management Interface") is your essential GPU dashboard — it lists each
GPU, its memory used/total, utilisation %, temperature, power, and the processes using it. Run
`watch -n 1 nvidia-smi` to refresh it once per second while a job runs, or install `nvitop` for a
nicer live view.

### Port forwarding (SSH tunnels): opening a remote Jupyter/TensorBoard in your local browser

Here's a problem you'll hit immediately: you start **JupyterLab** (or **TensorBoard**, or any web
UI) *on the pod*, listening on, say, port 8888 — but that port is on the *remote* machine, behind
its firewall. Your laptop's browser can't reach it directly. **SSH port forwarding** (a.k.a.
*tunnelling*) solves this elegantly: it carries traffic for a port *through* your existing encrypted
SSH connection, making a remote port appear as if it were running on your own laptop.

The form you'll use 95% of the time is **local forwarding** (`-L`):

```bash
ssh -L 8888:localhost:8888 root@213.181.x.x -p 41953 -i ~/.ssh/id_ed25519
```

Read `-L 8888:localhost:8888` as **"`-L local_port : host_reachable_from_the_pod : that_host's_port`"**:

- The **first** `8888` is a port on **your laptop**.
- `localhost:8888` is interpreted **from the pod's perspective** — "the machine the pod calls
  localhost (i.e. the pod itself), port 8888," where Jupyter is listening.
- Result: while that SSH session is open, visiting **`http://localhost:8888` in your laptop
  browser** transparently reaches Jupyter running on the pod, with all traffic encrypted inside the
  SSH tunnel. No public ports, no proxy needed.

You can forward several ports at once by repeating `-L` (e.g. add `-L 6006:localhost:6006` for
TensorBoard), and you can choose a different local port if 8888 is busy on your laptop
(`-L 8899:localhost:8888` → browse `localhost:8899`). With a `~/.ssh/config` alias (§10) you can
even bake forwards in with `LocalForward 8888 localhost:8888` so `ssh runpod` sets up the tunnel
automatically.

The mirror-image directions exist but you'll rarely need them: `-R` (**remote** forwarding) exposes
a *laptop* port to the *pod*, and `-D` opens a **dynamic** SOCKS proxy. For your work, `-L` is the
one to know cold. (Note: RunPod also gives you that `https://<pod-id>-8888.proxy.runpod.net` URL as
a no-tunnel alternative — but understanding `-L` means you can reach *any* remote port on *any*
server, not just RunPod's pre-proxied ones.)

---

## 16. Getting your code and data in, and results out

Four routes, from simplest to most powerful. All assume Option-A SSH for the transfer-capable ones.

### Git (best for code)

Your project is already a git repo, so the cleanest way to get code onto a pod is to clone it:

```bash
# on the pod:
cd /workspace
git clone https://github.com/<you>/<repo>.git
```

Edit on your laptop, `git push`, then `git pull` on the pod. Keeps everything versioned and avoids
copying. (For a private repo, use a token or add a *separate* SSH deploy key on the pod — don't copy
your laptop's private key onto a shared cloud machine.)

### `scp` — copy individual files/dirs over SSH

`scp` ("secure copy") uses the SSH connection to copy files. The syntax is
`scp [from] [to]`, where a remote location is written `user@host:path`:

```bash
# laptop → pod (upload a file):
scp -P 41953 -i ~/.ssh/id_ed25519 ./data.json root@213.181.x.x:/workspace/

# pod → laptop (download results):
scp -P 41953 -i ~/.ssh/id_ed25519 root@213.181.x.x:/workspace/results/out.npy ./

# a whole directory (recursive):
scp -r -P 41953 -i ~/.ssh/id_ed25519 ./mycode root@213.181.x.x:/workspace/
```

Note the capital **`-P`** for the port in `scp` (lowercase `-p` means "preserve timestamps" here —
an annoying inconsistency with `ssh`'s lowercase `-p`). If you set up `~/.ssh/config` with a `runpod`
alias, this all collapses to `scp ./data.json runpod:/workspace/` — the config supplies the port
and key automatically. That's a strong reason to do §10.

### `rsync` — efficient sync of large/changing trees (best for data + results)

`rsync` copies only the *differences* between source and destination, resumes interrupted transfers,
and is far better than `scp` for big or repeatedly-synced directories:

```bash
# upload a dataset, showing progress, only changed files:
rsync -avP -e "ssh -p 41953 -i ~/.ssh/id_ed25519" ./dataset/ root@213.181.x.x:/workspace/dataset/

# pull results back down:
rsync -avP -e "ssh -p 41953 -i ~/.ssh/id_ed25519" root@213.181.x.x:/workspace/results/ ./results/
```

- `-a` = archive (recursive, preserves metadata), `-v` = verbose, `-P` = progress + resume partial
  transfers, `-e "..."` tells rsync which ssh command (port/key) to tunnel through.
- Trailing slashes matter: `src/` means "the *contents* of src," `src` means "the src directory
  itself." With the `runpod` alias from §10 this shortens to
  `rsync -avP ./results/ runpod:/workspace/results/`.

### `runpodctl send` / `receive` — no SSH setup needed

The RunPod CLI ships a peer-to-peer transfer (built on `croc`) that works without configuring ports
or keys — handy with the proxy connection or for a one-off:

```bash
# on the sending side (e.g. the pod):
runpodctl send results.tar.gz
# → prints a one-time code like 8338-galileo-...

# on the receiving side (e.g. your laptop):
runpodctl receive 8338-galileo-...
```

For lots of small files, `tar` them first (`tar -czf results.tar.gz results/`) then send the single
archive — far faster than many separate transfers.

---

## 17. Cost hygiene: stop, terminate, and not getting surprised

This is the section that saves you money. The lifecycle of a pod has three states and the
distinction is financial:

| Action | GPU billing | Storage billing | Your data (`/workspace`) |
|---|---|---|---|
| **Running** | full hourly rate (per second) | included | present |
| **Stopped** | **none** | **small rate for the persistent volume** | **preserved** — restart to resume |
| **Terminated** | none | none | **gone forever** |

Key consequences:

- **When you finish for the day, STOP the pod** (not just close your SSH window). Closing the
  terminal does *nothing* to billing — the GPU keeps running and charging. Stop it from the console
  (or `runpodctl stop pod <id>`). A stopped pod keeps your `/workspace` so you can restart it later,
  but stops the expensive GPU meter; you still pay a little for the stored volume.
- **When you're truly done, TERMINATE** to stop *all* charges — but first pull your results off
  (§16), because termination deletes the volume disk. If you want data to persist across
  terminations, that's what a **network volume** is for (§12).
- **Watch your credit balance** for long runs — if credits hit zero mid-run the pod stops and
  in-progress work not yet written to disk is lost. Checkpoint long jobs to `/workspace` periodically.
- **Spot/interruptible pods** can be reclaimed at any time; only use them for work that checkpoints
  and can resume.
- A practical habit: note the hourly rate when you deploy and do the mental math — e.g. an A100 at
  ~$1.6–2.5/hr means a forgotten pod left running overnight (~10 hr) is ~$16–25. Not catastrophic,
  but it adds up across a thesis's worth of experiments.

`runpodctl` lets you script all of this from your laptop:

```bash
runpodctl get pod                 # list your pods and their status
runpodctl stop pod <pod-id>       # stop (keep volume, stop GPU billing)
runpodctl start pod <pod-id>      # restart a stopped pod
runpodctl remove pod <pod-id>     # terminate (deletes the pod) — check flag names with --help
```

---

## 18. End-to-end: a complete inference run on RunPod

Putting every piece together, here is the full loop for a research-inference run of a small
reasoning model — the workflow you'll actually repeat.

**On your laptop, once ever:**

```bash
ssh-keygen -t ed25519 -C "tonysu@macbook"          # §9 — make a key (skip if you have one)
ssh-add --apple-use-keychain ~/.ssh/id_ed25519     # §10 — cache passphrase in Keychain
cat ~/.ssh/id_ed25519.pub                           # §9 — copy this into RunPod Settings → SSH Keys
```

**Each session:**

```bash
# 1. Deploy a pod in the RunPod console (§14): pick a GPU, PyTorch template, sized /workspace.
#    Copy its Option-A SSH command from the Connect dialog.

# 2. (Optional) add/update the alias in ~/.ssh/config (§10) so you can type `ssh runpod`.

# 3. Connect from your laptop terminal:
ssh runpod                  # or the full: ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# 4. On the pod — the safety + sanity ritual:
tmux new -s infer           # §6 — survive disconnects. Do EVERYTHING below inside tmux.
nvidia-smi                  # §15 — confirm the GPU and its VRAM
cd /workspace               # §12 — work on the persistent volume

# 5. Get your code + environment:
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt        # the PyTorch template already has torch+CUDA working

# 6. (If needed) upload data from your laptop in ANOTHER local terminal tab:
#    rsync -avP ./data/ runpod:/workspace/<repo>/data/

# 7. Launch the run, logging everything, watching the GPU:
python run_inference.py > run.log 2>&1 &     # background, all output captured (§4, §6)
tail -f run.log                              # watch progress live; Ctrl-C just stops watching
watch -n 5 nvidia-smi                        # (in another tmux pane) keep an eye on GPU/VRAM

# 8. Detach and let it run unattended:
#    press Ctrl-b then d  →  you can now close your laptop; the job runs on the pod.
#    Reconnect later with:  ssh runpod   then   tmux attach -t infer

# 9. When the run finishes, pull results back to your laptop (local terminal):
rsync -avP runpod:/workspace/<repo>/results/ ./results/

# 10. STOP or TERMINATE the pod (§17) so you stop paying:
#     Console → Stop (keep volume) or Terminate (delete). Or: runpodctl stop pod <id>
```

That's the whole arc: keys made once, an encrypted SSH pipe to a rented GPU, `tmux` guarding the
job against disconnects, `rsync`/git moving bytes both ways, and disciplined stopping so the meter
isn't left running. Every command in it is something you now understand from first principles
rather than copy-pasting on faith.

---

## Appendix: a one-page command cheat-sheet

```text
NAVIGATION              FILES                      PIPES/REDIRECT
  pwd   where am I        cat / less   view          cmd > f     stdout→file (overwrite)
  ls -lah  list           head / tail  ends          cmd >> f    append
  cd dir   enter          tail -f      follow log     cmd 2>&1    merge stderr into stdout
  cd ..    up one          cp / mv / rm  copy/move/del cmd > f 2>&1  capture everything
  cd ~     home           mkdir -p     make dirs       a | b       pipe a's output into b
  cd -     previous        chmod 600 f  set perms       grep / wc / sort / find

PROCESSES & PERSISTENCE                   SSH
  ps aux | grep x   find a process          ssh-keygen -t ed25519          make keys
  top / htop / nvidia-smi   monitor         cat ~/.ssh/id_ed25519.pub      show PUBLIC key
  Ctrl-C abort   Ctrl-Z suspend   bg/fg     ssh user@host -p PORT -i KEY   connect
  cmd > log 2>&1 &   run in background      ssh alias                      via ~/.ssh/config
  tmux new -s name      start session       scp -P PORT -i KEY src dst     copy a file
  Ctrl-b d   detach                         rsync -avP -e "ssh ..." s d    sync a tree
  tmux attach -t name   reattach            ssh -L 8888:localhost:8888 …   tunnel a port

RUNPOD LIFECYCLE
  Deploy → Connect (copy SSH cmd) → tmux → work on /workspace → rsync results out → STOP/TERMINATE
  runpodctl get pod | start pod <id> | stop pod <id> | send <file> | receive <code>
  Container disk = scratch (wiped).  /workspace volume = persists on STOP, lost on TERMINATE.
```

*A companion to `GPU_AND_PARALLEL_COMPUTING_PRIMER.md` in this folder — that one explains what the
GPU is doing; this one explains how to reach it and drive it. Written 2026-06-28.*
