from pathlib import Path


def current_branch():
    head_dir = Path(".") / ".git" / "HEAD"
    print(head_dir)
    with head_dir.open("r") as f:
        content = f.read().splitlines()
    print(content)
    for line in content:
        print(line)
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


print(current_branch())
