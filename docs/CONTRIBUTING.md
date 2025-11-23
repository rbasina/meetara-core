# 🤝 Contributing to Meetara Core

Thank you for your interest in contributing to Meetara Core!

## Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/meetara-core.git
cd meetara-core
```

2. **Setup Environment**
```bash
python -m venv .venv-meetara
.venv-meetara\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **Configure**
```bash
cp env.example .env
# Edit .env with your configuration
```

4. **Run Tests**
```bash
pytest tests/
```

## Code Style

- **Formatting**: Black
- **Imports**: isort
- **Type Hints**: Required for all functions
- **Docstrings**: Required for all functions/classes

## Adding Features

### Adding a New Domain

1. Add domain to `config/domain_config.yaml`
2. Add keywords to `config/domain_keywords.yaml`
3. Test with document upload
4. **No code changes needed!**

### Adding a New Tool

1. Create tool in `app/agent/tools/your_tool.py`
2. Inherit from `BaseTool`
3. Implement `_run()` and `_arun()` methods
4. Add to agent in `app/agent/planner.py`

## Pull Request Process

1. Create feature branch
2. Make changes with tests
3. Update documentation if needed
4. Submit pull request with description

## Questions?

- Open an issue on GitHub
- Check `README.md` for documentation
- Review code examples in `tests/`

---

Thank you for contributing! 🎉

