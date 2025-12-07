# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of the Tail-Risk Hedge Lab seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security issues via email to:

**lorenzo.bassetti@gmail.com**

### What to Include

When reporting a vulnerability, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes (if available)
- Your contact information for follow-up

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 5 business days with our assessment
- **Resolution**: We aim to patch critical vulnerabilities within 30 days

### Disclosure Policy

- We request that you give us reasonable time to address the issue before public disclosure
- We will credit you for the discovery (unless you prefer to remain anonymous)
- We will provide updates on our progress toward a fix

## Security Best Practices for Users

When using this framework:

1. **Data Privacy**: Never commit sensitive financial data or API keys to version control
2. **Configuration Files**: Keep `config.yaml` out of public repositories if it contains proprietary settings
3. **Custom Data**: Excel files in `data/import/` may contain proprietary data - review before sharing
4. **API Keys**: If extending the framework to use paid data APIs, store credentials in environment variables
5. **Dependencies**: Regularly update dependencies via `pip install -r requirements.txt --upgrade`
6. **Output Reports**: HTML reports in `output/` may contain sensitive analysis - review before sharing

## Known Security Considerations

### Data Sources

- **yfinance**: Uses public market data from Yahoo Finance - no authentication required
- **FRED API**: Public U.S. economic data - no API key required for basic usage
- **Custom Excel Files**: User-provided data is loaded from `data/import/` - ensure source trustworthiness

### Code Execution

- The framework executes Python code for numerical analysis and optimization
- Custom Excel files are parsed using `pandas.read_excel()` - only load files from trusted sources
- No remote code execution or external script loading is performed

### Third-Party Dependencies

This project relies on several well-maintained Python packages:

- `pandas`, `numpy`: Data manipulation and numerical computing
- `yfinance`: Market data retrieval
- `scipy`: Statistical and optimization functions
- `matplotlib`: Visualization

We monitor security advisories for these dependencies and update as needed.

## Commercial Use Security

If using this framework in a commercial or institutional environment:

1. Review the [License](License.md) for commercial licensing requirements
2. Conduct your own security audit appropriate to your risk profile
3. Implement additional access controls for sensitive financial data
4. Consider running the framework in isolated environments for production use
5. Contact lorenzo.bassetti@gmail.com for commercial licensing and support

## Updates and Patches

Security updates will be:

- Released as soon as possible after discovery
- Announced via GitHub releases and commit messages
- Documented in release notes with severity assessment

## Questions?

For security-related questions that don't involve vulnerabilities, you can:

- Open a GitHub issue (for general security best practices)
- Contact lorenzo.bassetti@gmail.com for commercial/institutional security inquiries

---

**Last Updated**: December 7, 2025
