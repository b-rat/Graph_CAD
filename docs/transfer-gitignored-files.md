# Transfer Gitignored Files via Temporary Branch

Procedure for temporarily adding a gitignored file to a branch for RunPod transfer, then cleaning up.

## 1. Create a temporary branch

```bash
git checkout -b temp-runpod-transfer
```

## 2. Force-add the gitignored file

```bash
git add -f path/to/ignored/file.zip
```

The `-f` flag overrides `.gitignore`.

## 3. Commit and push

```bash
git commit -m "Temp: add file for RunPod transfer"
git push -u origin temp-runpod-transfer
```

## 4. On RunPod, clone/fetch the branch

```bash
# Fresh clone (single branch for speed):
git clone -b temp-runpod-transfer --single-branch https://github.com/user/repo.git

# Or if repo already exists:
git fetch origin temp-runpod-transfer
git checkout temp-runpod-transfer
```

## 5. Delete the branch (after transfer)

**Local:**

```bash
git checkout main
git branch -D temp-runpod-transfer
```

**Remote:**

```bash
git push origin --delete temp-runpod-transfer
```

## Alternative for Large Files

If the file is large, consider using `git lfs` or a direct transfer method like `scp`/`rsync` to avoid bloating git history. Even deleted branches can leave objects in the repo until garbage collection runs.
