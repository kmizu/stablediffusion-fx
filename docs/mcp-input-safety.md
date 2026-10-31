# Wildcard and image input safety

Wildcard lookup rejects parent components, absolute/UNC or drive-qualified
names. A candidate is resolved to a real path under its declared wildcard
directory before opening; an escaping symbolic link is rejected. The final
file is opened with `NOFOLLOW_LINKS`, and provenance hashes the same bytes
used for expansion. Normal nested wildcard names and glob behavior remain.
These checks do not provide an atomic filesystem sandbox: a concurrent change
to an ancestor directory or junction between resolution and opening is not
excluded, and resolving a junction to a network target may itself contact it.

Inline MCP image conversion removes its owned temporary PNG when the writer
throws or returns false. A filesystem that refuses deletion can still leave a
file; the cleanup exception is attached to the original failure. This is not a
claim that arbitrary external files can be removed.
