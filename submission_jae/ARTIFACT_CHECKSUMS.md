# Submission Artifact Checksums

Use these checksums to confirm that the uploaded files match the validated repository version.

## Manuscript PDF

| File | Size bytes | SHA256 |
|---|---:|---|
| `manuscript_jae/main.pdf` | 467092 | `7F0580FFCC2460D84B7DF2EB9D811CC9D2579BFD8E777EFCDD8BB167E8B812D3` |

## Verification Command

On Windows PowerShell:

```powershell
Get-FileHash manuscript_jae\main.pdf -Algorithm SHA256
```

The GitHub Actions workflow also validates the manuscript package on each push to `main`.
