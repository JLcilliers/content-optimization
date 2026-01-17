"""
SEO + AI Content Optimizer

A document-in, document-out SEO optimization tool that enhances DOCX content
and precisely highlights all new additions in green.

Critical Design Principle:
    Green highlighting integrity is the core value proposition.
    - Zero false positives (never highlight existing content)
    - Conservative approach: when uncertain, don't highlight

Example:
    >>> from seo_optimizer import optimize_document
    >>> result = optimize_document(
    ...     source_docx="input.docx",
    ...     keywords=["seo", "content"],
    ... )
    >>> result.save("output.docx")
"""

from importlib.metadata import version

__version__ = version("seo-optimizer")

# Core exports will be added as modules are implemented
__all__: list[str] = [
    "__version__",
]
