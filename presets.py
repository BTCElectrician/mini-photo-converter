"""
Image format presets - every common size in one place.

Categories:
- Social media (Twitter/X, Instagram, Facebook, LinkedIn, YouTube)
- Print at 300 DPI (postcards, flyers, business cards, posters)
- Web/app (buttons, icons, thumbnails, hero images, avatars)

Look up any preset with ``get_preset("banner")`` - names, aliases, and
shortcuts all resolve through the same table.
"""

from dataclasses import dataclass
from enum import Enum

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class FitMode(Enum):
    """How to handle images that don't match the target aspect ratio."""

    CROP = "crop"        # Fill the frame, trim overflow
    FIT = "fit"          # Fit inside, pad with background (letterbox)
    STRETCH = "stretch"  # Stretch to exact size (distorts)
    COVER = "cover"      # Same as CROP


@dataclass(frozen=True)
class Preset:
    """A named output format: dimensions plus how to get there."""

    name: str
    width: int
    height: int
    fit_mode: FitMode = FitMode.CROP
    output_folder: str = "output"
    suffix: str = ""
    dpi: int = 72  # 300 for print, 72 for screen
    background_color: tuple[int, int, int] = BLACK  # Letterbox fill
    description: str = ""

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


def _preset(name: str, width: int, height: int, folder: str, desc: str, *,
            fit: FitMode = FitMode.CROP, suffix: str | None = None,
            dpi: int = 72, bg: tuple[int, int, int] = BLACK) -> Preset:
    """Build a preset; the filename suffix defaults to ``_<name>``."""
    return Preset(name, width, height, fit, folder,
                  suffix if suffix is not None else f"_{name}", dpi, bg, desc)


def _print_preset(name: str, width: int, height: int, folder: str, desc: str, *,
                  fit: FitMode = FitMode.CROP, suffix: str | None = None) -> Preset:
    """A 300 DPI print preset; letterboxed formats pad with white."""
    return _preset(name, width, height, f"print/{folder}", desc,
                   fit=fit, suffix=suffix, dpi=300, bg=WHITE)


def _index(*presets: Preset) -> dict[str, Preset]:
    return {preset.name: preset for preset in presets}


SOCIAL_PRESETS = _index(
    _preset("twitter_banner", 1500, 500, "social/twitter", "Twitter/X header banner (3:1)"),
    _preset("twitter_banner_letterbox", 1500, 500, "social/twitter",
            "Twitter/X header banner with letterbox", fit=FitMode.FIT, suffix="_twitter_banner_lb"),
    _preset("twitter_post", 1200, 675, "social/twitter", "Twitter/X post image (16:9)"),
    _preset("instagram_post", 1080, 1080, "social/instagram", "Instagram square post (1:1)", suffix="_instagram"),
    _preset("instagram_portrait", 1080, 1350, "social/instagram", "Instagram portrait post (4:5)"),
    _preset("instagram_story", 1080, 1920, "social/instagram", "Instagram story (9:16)"),
    _preset("facebook_cover", 820, 312, "social/facebook", "Facebook cover photo", suffix="_fb_cover"),
    _preset("facebook_post", 1200, 630, "social/facebook", "Facebook post image", suffix="_fb_post"),
    _preset("linkedin_banner", 1584, 396, "social/linkedin", "LinkedIn profile banner"),
    _preset("linkedin_post", 1200, 627, "social/linkedin", "LinkedIn post image"),
    _preset("youtube_thumbnail", 1280, 720, "social/youtube", "YouTube thumbnail (16:9)", suffix="_yt_thumb"),
    _preset("youtube_banner", 2560, 1440, "social/youtube", "YouTube channel banner", suffix="_yt_banner"),
)

PRINT_PRESETS = _index(
    _print_preset("postcard", 1800, 1200, "postcards", "Postcard 6x4 inches (300 DPI)"),
    _print_preset("postcard_5x7", 2100, 1500, "postcards", "Postcard 7x5 inches (300 DPI)"),
    _print_preset("flyer_letter", 2550, 3300, "flyers", "Flyer letter size 8.5x11 (300 DPI)", fit=FitMode.FIT),
    _print_preset("flyer_a4", 2480, 3508, "flyers", "Flyer A4 size (300 DPI)", fit=FitMode.FIT),
    _print_preset("flyer_half", 1650, 2550, "flyers", "Half-page flyer 5.5x8.5 (300 DPI)", fit=FitMode.FIT),
    _print_preset("business_card", 1050, 600, "business_cards", "Business card 3.5x2 inches (300 DPI)", suffix="_bizcard"),
    _print_preset("poster_11x17", 3300, 5100, "posters", "Poster 11x17 inches (300 DPI)", fit=FitMode.FIT),
    _print_preset("poster_18x24", 5400, 7200, "posters", "Poster 18x24 inches (300 DPI)", fit=FitMode.FIT),
)

WEB_PRESETS = _index(
    _preset("button_small", 120, 40, "web/buttons", "Small button 120x40", fit=FitMode.FIT, suffix="_btn_sm"),
    _preset("button_medium", 200, 60, "web/buttons", "Medium button 200x60", fit=FitMode.FIT, suffix="_btn_md"),
    _preset("button_large", 300, 80, "web/buttons", "Large button 300x80", fit=FitMode.FIT, suffix="_btn_lg"),
    *(_preset(f"icon_{size}", size, size, "web/icons", f"Icon {size}x{size}")
      for size in (16, 32, 64, 128, 256, 512)),
    _preset("app_icon", 1024, 1024, "web/icons", "App icon 1024x1024 (iOS/Android)"),
    _preset("favicon", 32, 32, "web/icons", "Favicon 32x32"),
    _preset("avatar", 400, 400, "web/avatars", "Avatar / profile picture 400x400"),
    _preset("hero", 1920, 1080, "web/hero", "Hero image 1920x1080 (16:9)"),
    _preset("hero_wide", 2560, 600, "web/hero", "Wide hero banner 2560x600"),
    _preset("thumbnail_sm", 150, 150, "web/thumbnails", "Small thumbnail 150x150", suffix="_thumb_sm"),
    _preset("thumbnail_md", 300, 300, "web/thumbnails", "Medium thumbnail 300x300", suffix="_thumb_md"),
    _preset("thumbnail_lg", 600, 600, "web/thumbnails", "Large thumbnail 600x600", suffix="_thumb_lg"),
    _preset("og_image", 1200, 630, "web/og", "Open Graph image (link previews)", suffix="_og"),
)

# Friendly names -> canonical preset names.
ALIASES = {
    "banner": "twitter_banner",
    "button": "button_medium",
    "icon": "icon_256",
    "thumbnail": "thumbnail_md",
    "thumbnail_small": "thumbnail_sm",
    "thumbnail_medium": "thumbnail_md",
    "thumbnail_large": "thumbnail_lg",
    "postcard_4x6": "postcard",
    "flyer": "flyer_letter",
    "poster": "poster_11x17",
    "bizcard": "business_card",
    "instagram": "instagram_post",
    "instagram_square": "instagram_post",
    "og": "og_image",
}

_CANONICAL = {**SOCIAL_PRESETS, **PRINT_PRESETS, **WEB_PRESETS}

ALL_PRESETS: dict[str, Preset] = {
    **_CANONICAL,
    **{alias: _CANONICAL[target] for alias, target in ALIASES.items()},
}


def get_preset(name: str) -> Preset | None:
    """Look up a preset by name or alias (case- and separator-insensitive)."""
    return ALL_PRESETS.get(name.lower().replace("-", "_").replace(" ", "_"))


def create_custom_preset(name: str, width: int, height: int,
                         fit_mode: FitMode = FitMode.CROP,
                         output_folder: str = "custom",
                         background_color: tuple[int, int, int] = WHITE) -> Preset:
    """Create a one-off preset for an arbitrary size."""
    return Preset(name, width, height, fit_mode, output_folder,
                  suffix=f"_{name}", background_color=background_color,
                  description=f"Custom {width}x{height}")


def list_presets() -> None:
    """Print every available preset, grouped by category."""
    sections = [
        ("📱 SOCIAL MEDIA", SOCIAL_PRESETS),
        ("🖨️  PRINT (300 DPI)", PRINT_PRESETS),
        ("🌐 WEB/APP", WEB_PRESETS),
    ]

    print("\n" + "=" * 70)
    print("AVAILABLE PRESETS")
    print("=" * 70)

    for title, presets in sections:
        print(f"\n{title}:")
        print("-" * 40)
        for name, preset in presets.items():
            print(f"  {name:25} {preset.width}x{preset.height:5}  {preset.description}")

    print("\n⚡ SHORTCUTS:")
    print("-" * 40)
    for alias, target in ALIASES.items():
        print(f"  {alias:12}→ {target}")
    print()


if __name__ == "__main__":
    list_presets()
