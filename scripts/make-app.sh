#!/bin/sh
# Assemble Strouhal.app from the SwiftPM release build (ad-hoc signed).
# Ad-hoc signed; Developer ID and notarization need an Apple Developer account.
set -e
cd "$(dirname "$0")/.."
swift build -c release
APP=dist/Strouhal.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/StrouhalApp "$APP/Contents/MacOS/Strouhal"
cp -R .build/release/strouhal_StrouhalCore.bundle "$APP/Contents/Resources/"
cp Assets/AppIcon.icns "$APP/Contents/Resources/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleName</key><string>Strouhal</string>
    <key>CFBundleDisplayName</key><string>Strouhal</string>
    <key>CFBundleIdentifier</key><string>com.mandipadk.strouhal</string>
    <key>CFBundleExecutable</key><string>Strouhal</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleVersion</key><string>0.6.0</string>
    <key>CFBundleShortVersionString</key><string>0.6.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict></plist>
PLIST
codesign --force -s - "$APP"
echo "built $APP"
