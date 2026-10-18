#!/bin/sh
# Mahere's one-line installer for Linux and macOS (Nick 2026-10-17: Photon's, one paste and done):
#   curl -sSfL https://brobdingnagian.holdmyoscilloscope.com/mahere/install-release.sh | sh
# A file curl downloads carries no quarantine, so Gatekeeper never stops it; the binary checks its own signature before it is installed; after that Mahere updates itself.
set -e

BASE="https://brobdingnagian.holdmyoscilloscope.com/mahere"

echo "Mahere installer"
echo "================"
echo ""

OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
    Linux*)
        if [ "$ARCH" != "x86_64" ]; then
            echo "Error: Mahere for Linux is built for x86_64 only (this machine is $ARCH)."
            exit 1
        fi
        PLATFORM="linux-x86_64"
        ;;
    Darwin*)
        if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "aarch64" ]; then
            # Rosetta-translated shells report x86_64 on Apple Silicon: ask the hardware.
            if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]; then
                echo "Error: Mahere for Mac is built for Apple Silicon only (this Mac is Intel)."
                exit 1
            fi
        fi
        PLATFORM="macos-arm64"
        ;;
    *)
        echo "Error: unsupported system: $OS"
        exit 1
        ;;
esac

echo "Detected: $OS ($ARCH)"
echo ""

fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -sSfL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$1" -O "$2"
    else
        echo "Error: neither curl nor wget found."
        exit 1
    fi
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/mahere"

echo "Downloading Mahere..."
fetch "$BASE/mahere-$PLATFORM?v=$(date +%s)" "$BIN"
chmod +x "$BIN"
if [ "$OS" = "Darwin" ]; then
    xattr -c "$BIN" 2>/dev/null || true
fi

echo "Verifying signature..."
if ! "$BIN" verify >"$TMP/verify.log" 2>&1; then
    echo "Error: the downloaded Mahere did not pass its signature check:"
    cat "$TMP/verify.log"
    exit 1
fi
echo "✓ Signature verified"
echo ""

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

if [ "$OS" = "Darwin" ]; then
    # The binary lives in the app bundle (Finder, Dock, Spotlight, and the identity nz.mahere.app its Apple signature names); ~/.local/bin/mahere starts it, so the updater replaces the one copy.
    APP="$HOME/Applications/Mahere.app"
    echo "Installing $APP..."
    mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
    mv "$BIN" "$APP/Contents/MacOS/mahere"

    if fetch "$BASE/icon-1024.png" "$TMP/icon.png" 2>/dev/null; then
        mkdir -p "$TMP/icon.iconset"
        for s in 16 32 128 256 512; do
            sips -z $s $s "$TMP/icon.png" --out "$TMP/icon.iconset/icon_${s}x${s}.png" >/dev/null 2>&1 || true
            sips -z $((s * 2)) $((s * 2)) "$TMP/icon.png" --out "$TMP/icon.iconset/icon_${s}x${s}@2x.png" >/dev/null 2>&1 || true
        done
        iconutil -c icns "$TMP/icon.iconset" -o "$APP/Contents/Resources/AppIcon.icns" 2>/dev/null || true
    fi

    cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>mahere</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>nz.mahere.app</string>
    <key>CFBundleName</key>
    <string>Mahere</string>
    <key>CFBundleDisplayName</key>
    <string>Mahere</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST
    xattr -cr "$APP" 2>/dev/null || true
    # Finder picks up a new bundle's icon when the bundle is touched.
    touch "$APP"
    # A launcher, not a symlink: through a symlink the program's own path is the link's, and the updater would replace the link and leave the bundle behind.
    rm -f "$INSTALL_DIR/mahere"
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$APP/Contents/MacOS/mahere" > "$INSTALL_DIR/mahere"
    chmod +x "$INSTALL_DIR/mahere"
    echo "✓ Installed"
else
    echo "Installing to $INSTALL_DIR..."
    mv "$BIN" "$INSTALL_DIR/mahere"

    ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
    mkdir -p "$ICON_DIR"
    fetch "$BASE/icon-256.png" "$ICON_DIR/mahere.png" 2>/dev/null || true

    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    cat > "$DESKTOP_DIR/mahere.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Mahere
Comment=Offline map for hiking, biking and backroads
Exec=$INSTALL_DIR/mahere
Icon=mahere
Terminal=false
Categories=Education;Geography;Maps;
Keywords=map;topo;trail;hiking;offline;
EOF
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    echo "✓ Installed, with a menu entry"
fi

# ~/.local/bin on the PATH, in the shell's own startup file.
case "$SHELL" in
    */zsh) RC="$HOME/.zshrc" ;;
    */bash) if [ "$OS" = "Darwin" ]; then RC="$HOME/.bash_profile"; else RC="$HOME/.bashrc"; fi ;;
    */fish) RC="$HOME/.config/fish/config.fish" ;;
    *) RC="" ;;
esac
PATH_ADDED=""
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        if [ -n "$RC" ] && ! grep -q "$INSTALL_DIR" "$RC" 2>/dev/null; then
            mkdir -p "$(dirname "$RC")"
            printf '\n# Added by the Mahere installer\nexport PATH="$PATH:%s"\n' "$INSTALL_DIR" >> "$RC"
            PATH_ADDED="$RC"
        fi
        ;;
esac

echo ""
echo "✓ Mahere installed"
echo ""
if [ "$OS" = "Darwin" ]; then
    echo "Open it from ~/Applications, Spotlight or Launchpad (drag it to the Dock to keep it there),"
    echo "or run 'mahere' in Terminal."
else
    echo "Open it from your application menu, or run 'mahere'."
fi
if [ -n "$PATH_ADDED" ]; then
    echo "(For 'mahere' in Terminal, open a new window or run: . $PATH_ADDED)"
fi
echo "Mahere updates itself from here on: Settings → Updates."
