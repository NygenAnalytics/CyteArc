"""Render a small, inert Markdown subset for saved report narratives.

Only paragraphs, lists, emphasis, and inline code generate HTML. Links, images,
and supplied HTML remain text. Scientific identifiers and paths stay literal.
The Markdown companion escapes supplied text the same way, so no renderer
turns it into a link, image, heading, or HTML.
"""

import html
import re

# Relative Windows paths need another separator or a filename extension to
# distinguish them from prose containing literal newline escapes. A code span
# delimiter has at most 16 backticks: matching an unbounded delimiter
# backtracks cubically on a long run of backticks.
_LITERAL = re.compile(
    r"(?P<ticks>`{1,16})(?P<code>.*?)(?P=ticks)(?!`)"
    r"|(?P<path>(?<!\w)(?:[A-Za-z]:\\|\\\\|(?<!\S)\.\.?\\(?!n\\n)"
    r"|[\w.-]+\\(?=[^\s`<>\"']*(?:\\[^nr\\\s]|\.[a-zA-Z0-9]{1,8}\b)))"
    r"[^\s`<>\"']+)",
    re.DOTALL,
)
_INLINE = re.compile(
    _LITERAL.pattern + r"|(?P<posix>(?<!\w)(?:https?://|/|\.\.?/|[\w.-]+/)[^\s`<>\"']+)"
    r"|(?P<escaped>\\[\\`*_{}\[\]()#+.!>\-])"
    r"|(?P<marker>\*+|_+)|(?P<newline>\n)",
    re.DOTALL,
)
_LIST = re.compile(r"^( *)([-+*]|\d+[.)])\s+(.+)$")
# Backslashes, code and emphasis delimiters, link brackets, table pipes,
# headings and strikethrough marks; see markdown_text.
_MARKDOWN_SPECIAL = re.compile(r"[\\`*_\[\]|#~]")
# A line of only "-" or "=" would end a paragraph as a rule or heading.
_MARKDOWN_RULE = re.compile(r"^([ \t]*)([-=])(?=[-= \t]*$)", re.MULTILINE)


def markdown_text(text: str) -> str:
    """Write supplied text as inert Markdown for the report companion.

    Model and dataset text may contain Markdown links or images that a viewer
    would fetch, so every Markdown delimiter is backslash-escaped and HTML is
    entity-escaped. Without an unescaped ``[``, no link, image, or reference
    definition can form. Paragraphs and lists still render. Formatting the
    report itself adds, such as headings and its own links, is written
    around this text and stays active.
    """
    escaped = _MARKDOWN_SPECIAL.sub(r"\\\g<0>", text)
    return html.escape(_MARKDOWN_RULE.sub(r"\1\\\2", escaped))


def normalize_narrative(text: str) -> str:
    """Decode newline escapes in prose, preserving code, paths, and Unicode."""
    pieces = []
    start = 0
    for match in _LITERAL.finditer(text):
        pieces.append(
            text[start : match.start()].replace(r"\r\n", "\n").replace(r"\n", "\n")
        )
        pieces.append(match[0])
        start = match.end()
    pieces.append(text[start:].replace(r"\r\n", "\n").replace(r"\n", "\n"))
    return "".join(pieces).replace("\r\n", "\n").replace("\r", "\n")


def _inline(text: str) -> str:
    parts: list[str] = []
    opened: list[tuple[str, int]] = []
    start = 0
    tags = {
        1: ("<em>", "</em>"),
        2: ("<strong>", "</strong>"),
        3: ("<strong><em>", "</em></strong>"),
    }
    for match in _INLINE.finditer(text):
        parts.append(html.escape(text[start : match.start()]))
        start = match.end()
        token = match[0]
        if match["code"] is not None:
            parts.append("<code>" + html.escape(match["code"]) + "</code>")
        elif match["path"] or match["posix"]:
            parts.append(html.escape(token))
        elif match["escaped"]:
            parts.append(html.escape(token[1:]))
        elif match["newline"]:
            parts.append("<br>\n")
        else:
            before = text[match.start() - 1] if match.start() else " "
            after = text[match.end()] if match.end() < len(text) else " "
            can_close = not before.isspace() and not (after.isalnum() or after == "_")
            can_open = (
                not after.isspace()
                and after not in "*_"
                and not (before.isalnum() or before in "_*-./\\")
            )
            while (
                can_close
                and opened
                and opened[-1][0][0] == token[0]
                and len(opened[-1][0]) <= len(token)
            ):
                marker, index = opened.pop()
                opening, closing = tags[len(marker)]
                parts[index] = opening
                parts.append(closing)
                token = token[len(marker) :]
                if not token:
                    break
            if token:
                if can_open and len(token) in tags:
                    opened.append((token, len(parts)))
                parts.append(token)
    parts.append(html.escape(text[start:]))
    return "".join(parts)


def narrative_html(text: str) -> str:
    """Render bounded report prose as safe block elements, without dependencies."""
    lines = normalize_narrative(text).expandtabs(4).split("\n")
    output: list[str] = []
    paragraph: list[str] = []
    lists: list[tuple[int, str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            output.append("<p>" + _inline("\n".join(paragraph)) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        _, tag = lists.pop()
        output.append(f"</li></{tag}>")

    for line in lines:
        if not line.strip():
            flush_paragraph()
            while lists:
                close_list()
            continue
        item = _LIST.match(line)
        if item:
            flush_paragraph()
            indent, marker, body = item.groups()
            depth = len(indent)
            tag = "ol" if marker[0].isdigit() else "ul"
            while lists and (
                depth < lists[-1][0] or (depth == lists[-1][0] and tag != lists[-1][1])
            ):
                close_list()
            if lists and depth == lists[-1][0]:
                output.append("</li><li>")
            else:
                start = f' start="{int(marker[:-1])}"' if tag == "ol" else ""
                output.append(f"<{tag}{start}><li>")
                lists.append((depth, tag))
            output.append(_inline(body))
        elif lists and len(line) - len(line.lstrip()) > lists[-1][0]:
            output.append("<br>\n" + _inline(line.strip()))
        else:
            while lists:
                close_list()
            paragraph.append(line)
    flush_paragraph()
    while lists:
        close_list()
    return "".join(output)
