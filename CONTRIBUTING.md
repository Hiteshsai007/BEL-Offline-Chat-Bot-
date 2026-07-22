# Contributing to BEL Offline AI

Thank you for contributing to the BEL Offline AI Fault Code Assistant! Please follow these guidelines to keep our codebase clean, secure, and compatible with air-gapped systems.

---

## 1. Branching Strategy

To keep the `main` branch stable, we use a simple branching workflow:
1. **Never commit directly to `main`** (unless fixing minor documentation typos).
2. Create a new branch for your task:
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b bugfix/issue-description
   ```
3. Commit your changes locally with clear, descriptive commit messages.
4. Push your branch to GitHub and open a **Pull Request (PR)** to `main`.
5. Wait for the CI pipeline to pass and request a review from teammates before merging.

---

## 2. Windows & PowerShell Compatibility (Important!)

Because the system is designed to run on ship workstations which use default Windows PowerShell 5.1:
* **NO Unicode characters in PowerShell/batch scripts**: Never use special characters like boxes (`──`), smart quotes (`“”`), or em-dashes (`—`) in `.ps1` or `.bat` files. PowerShell 5.1 parses files as local ANSI (Windows-1252) by default and misinterprets these bytes as quotation marks, crashing the parser. Use standard ASCII dashes (`-` or `=`) instead.
* **Keep scripts portable**: Do not hardcode absolute paths like `C:\Users\...`. Use dynamic paths relative to the project root (e.g., `Split-Path -Parent $MyInvocation.MyCommand.Path` in PowerShell).

---

## 3. Local Development Flow

Before writing code, make sure you have run the setup script:
1. Double-click `First-Time-Setup.bat` (needs internet once to download models).
2. Use `.venv\Scripts\Activate.ps1` in PowerShell to activate the local environment.

### Code Style (Linter)
Our CI pipeline runs `flake8` to check for syntax issues and undefined variables. You should check your code locally before pushing:
```bash
# Install flake8 in your venv
pip install flake8

# Run the linter
flake8 app tests --max-line-length=120
```

### Running Tests
Always run the test suite locally to verify you did not break existing functionality:
```bash
python -m pytest tests/ -v
```

---

## 4. Pull Request Checklist

Before submitting your PR, ensure:
- [ ] Your code runs locally and does not require an active internet connection.
- [ ] All tests pass locally (`python -m pytest tests/ -v`).
- [ ] The linter returns no errors (`flake8 app tests`).
- [ ] You did not commit any sensitive keys or temporary logs (`logs/app.log` is ignored).
- [ ] Any UI changes look professional, responsive, and follow the design guidelines.
