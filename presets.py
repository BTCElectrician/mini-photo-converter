"""
Image Format Presets - All common sizes in one place

This module defines presets for converting images to various formats:
- Social Media (Twitter, Instagram, Facebook, LinkedIn, YouTube)
- Print (Postcards, Flyers, Business Cards, Posters)
- Web/App (Buttons, Icons, Thumbnails, Hero Images)
- Custom sizes

Each preset includes:
- Dimensions (width x height)
- Aspect ratio handling (crop, fit, stretch, letterbox)
- Output folder
- DPI (for print formats)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class FitMode(Enum):
    """How to handle images that don't match the target aspect ratio."""
    CROP = "crop"           # Crop to fill (lose edges)
    FIT = "fit"             # Fit inside, add padding (letterbox)
    STRETCH = "stretch"     # Stretch to fit (distort)
    COVER = "cover"         # Cover entire area, crop overflow


@dataclass
class Preset:
    """Image format preset definition."""
    name: str
    width: int
    height: int
    fit_mode: FitMode = FitMode.CROP
    output_folder: str = "output"
    suffix: str = ""
    dpi: int = 72  # For print: 300, for web: 72
    background_color: Tuple[int, int, int] = (0, 0, 0)  # For letterbox/fit mode
    description: str = ""

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


# =============================================================================
# SOCIAL MEDIA PRESETS
# =============================================================================

TWITTER_BANNER = Preset(
    name="twitter_banner",
    width=1500, height=500,
    fit_mode=FitMode.CROP,
    output_folder="social/twitter",
    suffix="_twitter_banner",
    description="Twitter/X Header Banner (3:1)"
)

TWITTER_BANNER_LETTERBOX = Preset(
    name="twitter_banner_letterbox",
    width=1500, height=500,
    fit_mode=FitMode.FIT,
    output_folder="social/twitter",
    suffix="_twitter_banner_lb",
    background_color=(0, 0, 0),
    description="Twitter/X Header Banner with letterbox"
)

TWITTER_POST = Preset(
    name="twitter_post",
    width=1200, height=675,
    fit_mode=FitMode.CROP,
    output_folder="social/twitter",
    suffix="_twitter_post",
    description="Twitter/X Post Image (16:9)"
)

INSTAGRAM_POST = Preset(
    name="instagram_post",
    width=1080, height=1080,
    fit_mode=FitMode.CROP,
    output_folder="social/instagram",
    suffix="_instagram",
    description="Instagram Square Post (1:1)"
)

INSTAGRAM_PORTRAIT = Preset(
    name="instagram_portrait",
    width=1080, height=1350,
    fit_mode=FitMode.CROP,
    output_folder="social/instagram",
    suffix="_instagram_portrait",
    description="Instagram Portrait Post (4:5)"
)

INSTAGRAM_STORY = Preset(
    name="instagram_story",
    width=1080, height=1920,
    fit_mode=FitMode.CROP,
    output_folder="social/instagram",
    suffix="_instagram_story",
    description="Instagram Story (9:16)"
)

FACEBOOK_COVER = Preset(
    name="facebook_cover",
    width=820, height=312,
    fit_mode=FitMode.CROP,
    output_folder="social/facebook",
    suffix="_fb_cover",
    description="Facebook Cover Photo"
)

FACEBOOK_POST = Preset(
    name="facebook_post",
    width=1200, height=630,
    fit_mode=FitMode.CROP,
    output_folder="social/facebook",
    suffix="_fb_post",
    description="Facebook Post Image"
)

LINKEDIN_BANNER = Preset(
    name="linkedin_banner",
    width=1584, height=396,
    fit_mode=FitMode.CROP,
    output_folder="social/linkedin",
    suffix="_linkedin_banner",
    description="LinkedIn Profile Banner"
)

LINKEDIN_POST = Preset(
    name="linkedin_post",
    width=1200, height=627,
    fit_mode=FitMode.CROP,
    output_folder="social/linkedin",
    suffix="_linkedin_post",
    description="LinkedIn Post Image"
)

YOUTUBE_THUMBNAIL = Preset(
    name="youtube_thumbnail",
    width=1280, height=720,
    fit_mode=FitMode.CROP,
    output_folder="social/youtube",
    suffix="_yt_thumb",
    description="YouTube Thumbnail (16:9)"
)

YOUTUBE_BANNER = Preset(
    name="youtube_banner",
    width=2560, height=1440,
    fit_mode=FitMode.CROP,
    output_folder="social/youtube",
    suffix="_yt_banner",
    description="YouTube Channel Banner"
)

# =============================================================================
# PRINT PRESETS (300 DPI)
# =============================================================================

POSTCARD_4X6 = Preset(
    name="postcard",
    width=1800, height=1200,  # 6x4 inches @ 300 DPI
    fit_mode=FitMode.CROP,
    output_folder="print/postcards",
    suffix="_postcard",
    dpi=300,
    description="Postcard 6x4 inches (300 DPI)"
)

POSTCARD_5X7 = Preset(
    name="postcard_5x7",
    width=2100, height=1500,  # 7x5 inches @ 300 DPI
    fit_mode=FitMode.CROP,
    output_folder="print/postcards",
    suffix="_postcard_5x7",
    dpi=300,
    description="Postcard 7x5 inches (300 DPI)"
)

FLYER_LETTER = Preset(
    name="flyer_letter",
    width=2550, height=3300,  # 8.5x11 inches @ 300 DPI
    fit_mode=FitMode.FIT,
    output_folder="print/flyers",
    suffix="_flyer_letter",
    dpi=300,
    background_color=(255, 255, 255),
    description="Flyer Letter Size 8.5x11 (300 DPI)"
)

FLYER_A4 = Preset(
    name="flyer_a4",
    width=2480, height=3508,  # A4 @ 300 DPI
    fit_mode=FitMode.FIT,
    output_folder="print/flyers",
    suffix="_flyer_a4",
    dpi=300,
    background_color=(255, 255, 255),
    description="Flyer A4 Size (300 DPI)"
)

FLYER_HALF = Preset(
    name="flyer_half",
    width=1650, height=2550,  # 5.5x8.5 inches @ 300 DPI
    fit_mode=FitMode.FIT,
    output_folder="print/flyers",
    suffix="_flyer_half",
    dpi=300,
    background_color=(255, 255, 255),
    description="Half-Page Flyer 5.5x8.5 (300 DPI)"
)

BUSINESS_CARD = Preset(
    name="business_card",
    width=1050, height=600,  # 3.5x2 inches @ 300 DPI
    fit_mode=FitMode.CROP,
    output_folder="print/business_cards",
    suffix="_bizcard",
    dpi=300,
    description="Business Card 3.5x2 inches (300 DPI)"
)

POSTER_11X17 = Preset(
    name="poster_11x17",
    width=3300, height=5100,  # 11x17 inches @ 300 DPI
    fit_mode=FitMode.FIT,
    output_folder="print/posters",
    suffix="_poster_11x17",
    dpi=300,
    background_color=(255, 255, 255),
    description="Poster 11x17 inches (300 DPI)"
)

POSTER_18X24 = Preset(
    name="poster_18x24",
    width=5400, height=7200,  # 18x24 inches @ 300 DPI
    fit_mode=FitMode.FIT,
    output_folder="print/posters",
    suffix="_poster_18x24",
    dpi=300,
    background_color=(255, 255, 255),
    description="Poster 18x24 inches (300 DPI)"
)

# =============================================================================
# WEB/APP PRESETS
# =============================================================================

BUTTON_SMALL = Preset(
    name="button_small",
    width=120, height=40,
    fit_mode=FitMode.FIT,
    output_folder="web/buttons",
    suffix="_btn_sm",
    description="Small Button 120x40"
)

BUTTON_MEDIUM = Preset(
    name="button_medium",
    width=200, height=60,
    fit_mode=FitMode.FIT,
    output_folder="web/buttons",
    suffix="_btn_md",
    description="Medium Button 200x60"
)

BUTTON_LARGE = Preset(
    name="button_large",
    width=300, height=80,
    fit_mode=FitMode.FIT,
    output_folder="web/buttons",
    suffix="_btn_lg",
    description="Large Button 300x80"
)

ICON_16 = Preset(
    name="icon_16",
    width=16, height=16,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_icon_16",
    description="Icon 16x16 (favicon size)"
)

ICON_32 = Preset(
    name="icon_32",
    width=32, height=32,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_icon_32",
    description="Icon 32x32 (favicon)"
)

ICON_64 = Preset(
    name="icon_64",
    width=64, height=64,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_icon_64",
    description="Icon 64x64"
)

ICON_128 = Preset(
    name="icon_128",
    width=128, height=128,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_icon_128",
    description="Icon 128x128"
)

ICON_256 = Preset(
    name="icon_256",
    width=256, height=256,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_icon_256",
    description="Icon 256x256"
)

ICON_512 = Preset(
    name="icon_512",
    width=512, height=512,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_icon_512",
    description="Icon 512x512"
)

APP_ICON = Preset(
    name="app_icon",
    width=1024, height=1024,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_app_icon",
    description="App Icon 1024x1024 (iOS/Android)"
)

FAVICON = Preset(
    name="favicon",
    width=32, height=32,
    fit_mode=FitMode.CROP,
    output_folder="web/icons",
    suffix="_favicon",
    description="Favicon 32x32"
)

HERO_IMAGE = Preset(
    name="hero",
    width=1920, height=1080,
    fit_mode=FitMode.CROP,
    output_folder="web/hero",
    suffix="_hero",
    description="Hero Image 1920x1080 (16:9)"
)

HERO_WIDE = Preset(
    name="hero_wide",
    width=2560, height=600,
    fit_mode=FitMode.CROP,
    output_folder="web/hero",
    suffix="_hero_wide",
    description="Wide Hero Banner 2560x600"
)

THUMBNAIL_SM = Preset(
    name="thumbnail_sm",
    width=150, height=150,
    fit_mode=FitMode.CROP,
    output_folder="web/thumbnails",
    suffix="_thumb_sm",
    description="Small Thumbnail 150x150"
)

THUMBNAIL_MD = Preset(
    name="thumbnail_md",
    width=300, height=300,
    fit_mode=FitMode.CROP,
    output_folder="web/thumbnails",
    suffix="_thumb_md",
    description="Medium Thumbnail 300x300"
)

THUMBNAIL_LG = Preset(
    name="thumbnail_lg",
    width=600, height=600,
    fit_mode=FitMode.CROP,
    output_folder="web/thumbnails",
    suffix="_thumb_lg",
    description="Large Thumbnail 600x600"
)

OG_IMAGE = Preset(
    name="og_image",
    width=1200, height=630,
    fit_mode=FitMode.CROP,
    output_folder="web/og",
    suffix="_og",
    description="Open Graph Image (link previews)"
)

# =============================================================================
# PRESET COLLECTIONS
# =============================================================================

SOCIAL_PRESETS = {
    "twitter_banner": TWITTER_BANNER,
    "twitter_banner_letterbox": TWITTER_BANNER_LETTERBOX,
    "twitter_post": TWITTER_POST,
    "instagram": INSTAGRAM_POST,
    "instagram_post": INSTAGRAM_POST,
    "instagram_portrait": INSTAGRAM_PORTRAIT,
    "instagram_story": INSTAGRAM_STORY,
    "facebook_cover": FACEBOOK_COVER,
    "facebook_post": FACEBOOK_POST,
    "linkedin_banner": LINKEDIN_BANNER,
    "linkedin_post": LINKEDIN_POST,
    "youtube_thumbnail": YOUTUBE_THUMBNAIL,
    "youtube_banner": YOUTUBE_BANNER,
}

PRINT_PRESETS = {
    "postcard": POSTCARD_4X6,
    "postcard_4x6": POSTCARD_4X6,
    "postcard_5x7": POSTCARD_5X7,
    "flyer": FLYER_LETTER,
    "flyer_letter": FLYER_LETTER,
    "flyer_a4": FLYER_A4,
    "flyer_half": FLYER_HALF,
    "business_card": BUSINESS_CARD,
    "bizcard": BUSINESS_CARD,
    "poster": POSTER_11X17,
    "poster_11x17": POSTER_11X17,
    "poster_18x24": POSTER_18X24,
}

WEB_PRESETS = {
    "button": BUTTON_MEDIUM,
    "button_small": BUTTON_SMALL,
    "button_medium": BUTTON_MEDIUM,
    "button_large": BUTTON_LARGE,
    "icon": ICON_256,
    "icon_16": ICON_16,
    "icon_32": ICON_32,
    "icon_64": ICON_64,
    "icon_128": ICON_128,
    "icon_256": ICON_256,
    "icon_512": ICON_512,
    "app_icon": APP_ICON,
    "favicon": FAVICON,
    "hero": HERO_IMAGE,
    "hero_wide": HERO_WIDE,
    "thumbnail": THUMBNAIL_MD,
    "thumbnail_small": THUMBNAIL_SM,
    "thumbnail_medium": THUMBNAIL_MD,
    "thumbnail_large": THUMBNAIL_LG,
    "og": OG_IMAGE,
    "og_image": OG_IMAGE,
}

# Combined lookup - shortcuts for common uses
ALL_PRESETS = {
    # Shortcuts (most common)
    "banner": TWITTER_BANNER,
    "button": BUTTON_MEDIUM,
    "icon": ICON_256,
    "thumbnail": THUMBNAIL_MD,
    "postcard": POSTCARD_4X6,
    "flyer": FLYER_LETTER,
    "poster": POSTER_11X17,
    "hero": HERO_IMAGE,

    **SOCIAL_PRESETS,
    **PRINT_PRESETS,
    **WEB_PRESETS,
}


def get_preset(name: str) -> Optional[Preset]:
    """Get a preset by name."""
    return ALL_PRESETS.get(name.lower().replace("-", "_").replace(" ", "_"))


def list_presets() -> None:
    """Print all available presets."""
    print("\n" + "=" * 70)
    print("AVAILABLE PRESETS")
    print("=" * 70)

    print("\n📱 SOCIAL MEDIA:")
    print("-" * 40)
    for name, preset in SOCIAL_PRESETS.items():
        print(f"  {name:25} {preset.width}x{preset.height:5}  {preset.description}")

    print("\n🖨️  PRINT (300 DPI):")
    print("-" * 40)
    for name, preset in PRINT_PRESETS.items():
        print(f"  {name:25} {preset.width}x{preset.height:5}  {preset.description}")

    print("\n🌐 WEB/APP:")
    print("-" * 40)
    for name, preset in WEB_PRESETS.items():
        print(f"  {name:25} {preset.width}x{preset.height:5}  {preset.description}")

    print("\n⚡ SHORTCUTS:")
    print("-" * 40)
    print("  banner      → twitter_banner")
    print("  button      → button_medium")
    print("  icon        → icon_256")
    print("  thumbnail   → thumbnail_medium")
    print("  postcard    → postcard_4x6")
    print("  flyer       → flyer_letter")
    print("  poster      → poster_11x17")
    print("  hero        → hero (1920x1080)")
    print()


def create_custom_preset(
    name: str,
    width: int,
    height: int,
    fit_mode: FitMode = FitMode.CROP,
    output_folder: str = "custom",
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Preset:
    """Create a custom preset."""
    return Preset(
        name=name,
        width=width,
        height=height,
        fit_mode=fit_mode,
        output_folder=output_folder,
        suffix=f"_{name}",
        background_color=background_color,
        description=f"Custom {width}x{height}"
    )


if __name__ == "__main__":
    list_presets()
