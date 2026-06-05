#!/usr/bin/env python3
"""
Blog quality validator for LearningLab blogs.

Usage:
    python scripts/validate-blog.py <path-to-blog.md>

Example:
    python scripts/validate-blog.py 06-rag/blog-rag.md

Checks:
    - All inline citations have a References section entry
    - All HuggingFace dataset links are properly formatted
    - All code blocks have a language tag
    - Required sections are present
    - No section is bullet-list-only (has at least one paragraph)
    - Word count per section is within target range
    - Audience callouts (Interview/Production/Go deeper) are present
    - Resources boxes appear after major sections
"""

import re
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_sections(text):
    """Return list of (heading, body) tuples."""
    pattern = re.compile(r'^(#{1,3} .+)$', re.MULTILINE)
    headings = [(m.start(), m.group()) for m in pattern.finditer(text)]
    sections = []
    for i, (pos, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        body = text[pos:end]
        sections.append((heading.strip(), body))
    return sections


def word_count(text):
    return len(text.split())


# ── checks ───────────────────────────────────────────────────────────────────

def check_inline_citations_have_references(text):
    """Every (Author, YEAR) inline citation must appear in the References section."""
    issues = []
    inline = re.findall(r'\[([^\]]+,\s*\d{4}[a-z]?)\]\(https?://[^\)]+\)', text)

    refs_match = re.search(r'## References(.+)', text, re.DOTALL)
    refs_text = refs_match.group(1) if refs_match else ""

    for cite in set(inline):
        # Extract just the year from "Author et al., 2023"
        year_match = re.search(r'(\d{4})', cite)
        if year_match and year_match.group(1) not in refs_text:
            issues.append(f"  ⚠ Inline citation '{cite}' has no matching References entry")
    return issues


def check_hf_dataset_links(text):
    """HuggingFace dataset links should use the standard format."""
    issues = []
    # Find all HF dataset links — should be huggingface.co/datasets/
    raw_hf = re.findall(r'huggingface\.co/(?!datasets/|models/|spaces/|blog/)(\S+)', text)
    for link in raw_hf:
        issues.append(f"  ⚠ Possible unformatted HF link: huggingface.co/{link} — should be huggingface.co/datasets/ or /models/")
    return issues


def check_code_blocks_have_language(text):
    """Code blocks should specify a language."""
    issues = []
    bare_blocks = re.findall(r'\n```\n', text)
    if bare_blocks:
        issues.append(f"  ⚠ {len(bare_blocks)} code block(s) missing language tag (use ```python, ```bash, etc.)")
    return issues


def check_required_sections(text):
    """Check that high-value sections are present."""
    issues = []
    required = [
        ("References", r'## References'),
        ("The Problem / Introduction", r'## \d+\. The Problem|## 1\.|## Introduction'),
        ("History section", r'## \d+\. .*(History|Evolution|Background|From )'),
        ("Code example", r'```python'),
        ("Audience callouts", r'🎯|🏭|📚'),
    ]
    for name, pattern in required:
        if not re.search(pattern, text, re.IGNORECASE):
            issues.append(f"  ✗ Missing: {name}")
    return issues


def check_no_bullet_only_sections(text):
    """Each section should have at least one paragraph (not just bullets)."""
    issues = []
    sections = extract_sections(text)
    for heading, body in sections:
        lines = [l.strip() for l in body.split('\n') if l.strip()]
        content_lines = [l for l in lines if not l.startswith('#')]
        if not content_lines:
            continue
        has_paragraph = any(
            not l.startswith(('-', '*', '|', '>', '`', '#', '!'))
            for l in content_lines
        )
        if not has_paragraph:
            issues.append(f"  ⚠ Section '{heading}' appears to be bullet-list-only — add narrative prose")
    return issues


def check_word_counts(text):
    """Warn if total word count is outside expected range."""
    issues = []
    total = word_count(text)
    if total < 2000:
        issues.append(f"  ⚠ Blog is very short ({total} words) — target is 4,000–8,000 words")
    elif total > 12000:
        issues.append(f"  ℹ Blog is long ({total} words) — consider splitting into parts")
    else:
        issues.append(f"  ✓ Word count: {total} words (target: 4,000–8,000)")
    return issues


def check_resources_boxes(text):
    """Major sections should end with a Resources box."""
    issues = []
    major_sections = re.findall(r'^## .+', text, re.MULTILINE)
    resources_count = len(re.findall(r'\*\*Resources\*\*', text))
    if len(major_sections) > 3 and resources_count < 2:
        issues.append(f"  ⚠ Only {resources_count} Resources box(es) found for {len(major_sections)} sections — add 'Resources' boxes at the end of major sections")
    return issues


def check_formulas_have_examples(text):
    """Formulas (code blocks with math) should be followed by a worked example."""
    issues = []
    formula_blocks = re.finditer(r'```\n([^`]+(?:FORMULA|formula|loss\s*=|score\s*=|\bΣ\b|\bnDCG\b)[^`]+)```', text)
    for match in formula_blocks:
        # Check if a worked example follows within ~500 characters
        following = text[match.end():match.end() + 600]
        if not re.search(r'[Ee]xample|[Ss]tep \d|=\s*[\d\.]', following):
            issues.append("  ⚠ Formula block found without a worked example immediately after it")
    return issues


# ── main ─────────────────────────────────────────────────────────────────────

def validate(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    text = path.read_text(encoding='utf-8')
    print(f"\nValidating: {filepath}")
    print("=" * 60)

    all_issues = []
    checks = [
        ("Citations",          check_inline_citations_have_references),
        ("HuggingFace links",  check_hf_dataset_links),
        ("Code block tags",    check_code_blocks_have_language),
        ("Required sections",  check_required_sections),
        ("Prose vs bullets",   check_no_bullet_only_sections),
        ("Word count",         check_word_counts),
        ("Resources boxes",    check_resources_boxes),
        ("Formula examples",   check_formulas_have_examples),
    ]

    for name, fn in checks:
        issues = fn(text)
        print(f"\n[{name}]")
        if issues:
            for issue in issues:
                print(issue)
            all_issues.extend(issues)
        else:
            print(f"  ✓ Passed")

    print("\n" + "=" * 60)
    errors   = [i for i in all_issues if i.strip().startswith("✗")]
    warnings = [i for i in all_issues if i.strip().startswith("⚠")]
    print(f"Result: {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        print("\nFix errors before publishing:")
        for e in errors:
            print(e)
        sys.exit(1)
    elif warnings:
        print("\nConsider addressing warnings:")
        for w in warnings:
            print(w)
    else:
        print("Blog looks good!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate-blog.py <path-to-blog.md>")
        sys.exit(1)
    validate(sys.argv[1])
