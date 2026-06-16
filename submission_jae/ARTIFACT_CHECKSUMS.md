# Submission Artifact Checksums

Use these checksums to confirm that the uploaded files match the validated repository version.

## Manuscript PDF

| File | Size bytes | SHA256 |
|---|---:|---|
| `manuscript_jae/main.pdf` | 465434 | `AC5DBBEDB40CDA1726204CF65B5258EBADE1CB140B43DA8C4B26D37FD1176E69` |

## Verification Command

On Windows PowerShell:

```powershell
Get-FileHash manuscript_jae\main.pdf -Algorithm SHA256
```

The GitHub Actions workflow also validates the manuscript package on each push to `main`.
