"""Constants shared across modules. Imports nothing of ours, so any module may import it."""

# Camera raw formats LibRaw decodes.
LIBRAW_EXTENSIONS = frozenset({
    ".3fr", ".ari", ".arw", ".bay", ".cap", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf", ".fff", ".gpr", ".iiq",
    ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".ori", ".pef", ".raf", ".raw", ".rw2",
    ".rwl", ".sr2", ".srf", ".srw", ".x3f",
})

# Still image formats, including all still raw formats.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fit", ".fits"}) | LIBRAW_EXTENSIONS

# Video formats.
VIDEO_EXTENSIONS = frozenset({
    ".3g2", ".3gp", ".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".ogv", ".ser", ".webm", ".wmv",
})
