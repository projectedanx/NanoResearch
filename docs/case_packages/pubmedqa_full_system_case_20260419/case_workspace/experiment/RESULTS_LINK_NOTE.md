原始 workspace 中，`experiment/results` 是一个软链接：

- `experiment/results -> <workspace>/results`

在这个 case 包里，真实结果已经保留在上层目录：

- `case_workspace/results/`

因此这里不再保留软链接，避免 GitHub 上出现断链。
