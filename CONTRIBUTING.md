# Contributing to Modern Indian ALPR

First, thank you for your interest in contributing to Modern Indian ALPR! We appreciate all contributions, whether they're bug fixes, new features, or documentation improvements.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building a welcoming community for everyone.

## How to Contribute

### Reporting Bugs

- **Check existing issues first** - Search the issue tracker to see if the bug has already been reported
- **Provide detailed information**:
  - Python version and OS (Windows/Linux/macOS)
  - GPU/CPU setup (if GPU used, which model)
  - Steps to reproduce
  - Expected vs actual behavior
  - Error messages and stack traces
  - Screenshots/videos if applicable

### Suggesting Enhancements

- **Check existing discussions** - Avoid duplicate feature requests
- **Describe the use case** - Explain why you need this feature
- **Provide examples** - Show how it would work
- **Consider performance** - Discuss potential impact on speed/memory

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SIH-2026.git
   cd SIH-2026
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style and patterns
   - Add comments for complex logic
   - Update relevant documentation

4. **Test thoroughly**
   ```bash
   # Test on images
   python modern_alpr.py --source test_image.jpg --output results/

   # Test on videos
   python modern_alpr.py --source test_video.mp4 --output results/ --max-frames 50

   # If modifying core modules, test both CPU and GPU
   python modern_alpr.py --device cpu --source test_image.jpg
   python modern_alpr.py --device cuda --source test_image.jpg
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "feat: add feature description" 
   # or
   git commit -m "fix: fix bug description"
   git commit -m "docs: update documentation"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Title: Clear, concise description
   - Description: Explain changes and why they're needed
   - Link related issues if applicable

## Development Setup

### Environment Setup
```powershell
# Clone and setup
git clone https://github.com/Grim-235/SIH-2026.git
cd SIH-2026

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Running Tests
```powershell
# Test basic CLI
python modern_alpr.py --source inputs/ --output results/test/

# Test with different device
python modern_alpr.py --device cpu --source inputs/

# Test dashboard (requires streamlit)
streamlit run dashboard.py

# Test web API (requires flask)
python app.py
```

## Coding Guidelines

### Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Keep functions focused and modular
- Max line length: 120 characters

### Comments
- Add docstrings to functions and classes
- Explain "why" not just "what"
- Include examples for complex functionality

### Error Handling
```python
# Good: Specific error handling
try:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")
except RuntimeError as e:
    logger.error(f"Image loading failed: {e}")
    raise

# Avoid: Generic exception catching
try:
    frame = cv2.imread(str(source))
except:
    pass
```

### Performance Considerations
- Use GPU acceleration where appropriate
- Minimize data copying between CPU/GPU
- Profile code for bottlenecks
- Document performance characteristics

## Documentation

### README Updates
- Keep setup instructions current
- Document new features with examples
- Update troubleshooting section if needed

### Code Comments
- Add docstrings with type hints
- Explain non-obvious logic
- Include usage examples for public functions

### Examples
- Create example scripts if adding major features
- Test examples before committing

## Version Control

### Branch Naming
- Features: `feature/short-description`
- Bug fixes: `fix/short-description`
- Documentation: `docs/short-description`
- Experiments: `exp/short-description`

### Commit Messages
Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Adding/updating tests

Example:
```
feat: add vehicle type detection

- Implement CNN-based vehicle classifier
- Add vehicle type to detection output
- Update dashboard to show type statistics
- Closes #42
```

## Issues & Discussions

### Creating Issues
- Use bug reports for bugs
- Use feature requests for enhancements
- Use discussions for questions
- Provide reproducible examples

### Responding to Issues
- Be courteous and helpful
- Provide working solutions
- Link relevant resources
- Follow up on fixes

## Review Process

When your PR is submitted:
1. Automated checks will run (if configured)
2. Maintainers will review the code
3. Feedback will be provided
4. Make requested changes
5. Once approved, it will be merged!

## Questions?

- Open an issue with your question
- Start a discussion thread
- Check existing documentation

## Acknowledgments

Thank you for contributing to Modern Indian ALPR! Your efforts help make this project better for everyone. All contributors will be recognized in the project README.

---

Happy contributing! 🚀
