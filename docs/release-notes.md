# Mahere release notes

## 0.225 · October 2026

### Faster on older phones
- The map draws several times faster on phones like the Pixel 3a: each part of the screen now looks up its map tiles once instead of at every pixel.
- Map layers that don't change, like land, water, ownership and imagery, are combined once per tile, while the hillshade, slope bands, contours and every road and trail still draw live.
- Only the layers you have switched on are drawn; the rest cost nothing.
- A still view sharpens as far as the phone can manage instead of always aiming for the sharpest.

### Fixes
- Older phones no longer show "not responding" or stall while a burst of new map detail arrives.
- The map no longer turns black after the graphics driver resets: it rebuilds itself and carries on.
- Old, coarse tiles no longer flash over sharper ones that have already loaded, and no more brown patches where detail is still on its way.
- Roads fade against the biggest road actually on screen, not one just out of view.

## 0.224 · October 2026

### Notes
- An eraser: pick it from the toolbox, or flip the S Pen to its eraser end or hold its side button. It cleans up ink at every zoom and leaves the rest; Clear ink sits with it for the rare fresh start.

### Menu
- Each layer's own colours and settings open right under it: tap the … at the end of a row, then a setting to adjust its sliders in place. No more separate pages.
- Contours have an Interval slider from 1 cm to 1 km; moving it holds that spacing, and Auto interval fits each view again.

### Lettering
- The glow around map names is softer on every theme, and follows Linear highlights instead of staying bright white.

## 0.223 · October 2026

### Faster to open, and it remembers
- Mahere opens in a few seconds instead of up to a minute: the map's downloads are kept as plain files, apart from your own data.
- Downloaded detail stays downloaded instead of being fetched again each time.
- After it updates itself, Mahere opens again where you were, with a notice to tap if it can't.

### On the map
- The seabed of Puget Sound around Vashon, Elliott Bay and the rest shows its depths, no longer a flat sheet of water.
- A new layer, Sea level (geoid): how far sea level sits above or below the GPS's ellipsoid, coloured, with its own contour lines.
- Labels appear on the whole globe again, and an old layout no longer piles up in one corner while you zoom.
- Chinese names draw in a Chinese typeface even with Japan or Korea in view, without boxes.

### Menu
- The layers fold into groups you open and close: Ground, Imagery, Lines and names, Land and water, Light, Location.
- Contours have an Auto interval switch: turn it off to keep the spacing as you move.
- Turning on imagery greys out the terrain and tint, which don't show under it.

### Fixes
- Tapping the map no longer freezes the phone for several seconds.
- Zoomed out over South America and elsewhere, the map no longer shows bare patches.

## 0.21x · October 2026

### On your computer
- Mahere for Mac (Apple Silicon) and Windows, beside Linux, updating itself like the phone.
- One line to install on Mac, Linux and Windows: paste it into Terminal or PowerShell and press Enter.
- The map is drawn on your graphics card, as on the phone: much faster, with smooth contour lines.
- The map grows and shrinks with the window; Ctrl+scroll (Cmd+scroll on a Mac) sizes the controls and text.
- The compass and zoom sit below the window buttons instead of under them.

### Lettering
- Names and the route are drawn into the map's own detailed layer and smoothed with it, so letters and lines scale together as one picture.

### More on the map
- Freeway exits with their numbers and names, and ramps that say where they go ("toward Seneca St"), in directions too.
- Islands, bays, straits, capes, neighbourhoods and named backcountry places.
- Ferries across the water, which directions can take; ski lifts; named ridges and cliffs.
- Fire lookouts, hot springs, caves, shelters, ranger stations, drinking water, toilets, gates and fords, each with its own mark.
- Directions know how hard a trail is and how rough a road is: walking slows on scrambles, bikes stay off alpine routes, cars keep to roads they can drive.
- Puget Sound and the rest of the sea in the Seattle area are water again, not land.

### Fixes
- Changing the phone's screen resolution no longer makes the app crash and restart over and over.
- A place's name shows once, not twice side by side.
- The zoom number under the compass turns amber when you zoom past the map's detail, white while there is more to come.
- The sea no longer turns to land in patches after the app recovers its graphics.

### Names in their own scripts
- Place names everywhere read as the people there write them: 北京市, 東京都, 서울, القاهرة, תל אביב, Москва.
- The fonts for each script download the first time you look at a place that needs one, then stay on the phone; the app itself stays small.
- Japanese names get Japanese letterforms and Korean names Korean, not one shared set.
- Arabic, Hebrew and the Indian scripts show in full on the info card; on the map itself they arrive in a later update.

### Faster maps
- The map's cells are now held at Cloudflare's servers near you, so a cell anyone has viewed before arrives about three times sooner.
- A new view fills in about twice as fast: lines in under two seconds, then terrain.
- Once the map is still, the cells just around it and one zoom level closer download quietly, so panning and zooming in show from the phone at once.
- Connections to the map server are kept open between cells rather than made again each time.

### Fixes
- Large-screen phones (1440 by 3040, like the Note 10+) no longer flash black.
- River names no longer show inside lakes.
- Automatic updates install only a newer release, never an older one or a test build.

## 0.20x · October 2026

### A new map format
- Every cell is about a fifth smaller, with nothing lost: the same terrain, imagery and lines, coded more tightly.
- Imagery and infrared download only when you turn them on, so the map without them is about a quarter of the data in the 1 m areas.
- Downloaded areas keep their imagery for offline use. When the phone's cache is full, imagery is let go before the map itself.
- Older versions of the app can't read the new map: install this one fresh.

### Lettering
- Each map style has its own typeface: Oxanium for Neon, a classic serif for Atlas and Gazetteer, carved capitals for Woodcut, a draughtsman's mono for Blueprint, and more, with colours and glow to match. Pick any typeface for any style in the style editor.
- Names are weighted by importance among what's on screen: the biggest town in view is the boldest and largest, the smallest the lightest; peaks by height, rivers by size, lakes by area.
- A lake's name follows the lake's own shape, curving along its length.

### Tools
- A toolbox next to the locate button: Info, Ruler, Drive, Ride, Walk, Pen and Select, with the pen's colour and the clears beside their tools.
- A plain tap now shows a small card with what's there and its coordinates, without the distance line and profile; the Ruler measures as before.
- Directions start where the ruler does: the middle of the screen, or you with "Measure from me" on.
- A scale bar, and a button that takes the map to where you are.
- Every settings page has a way back.

### New detail
- Snoqualmie Pass to Leavenworth now has directions too.

## 0.19x · October 2026

### Names on the map
- Places, peaks, lakes, rivers, roads, trails and parks are lettered on the map, in the order a map reader looks for them: towns and summits first, then lakes, highways, rivers, villages, trails and streets.
- Labels follow the map smoothly as you pan and turn it. Names along roads and rivers bend with them and always read the right way up.
- Tap anything to hear what it is: the road or trail and its number, the nearest peak or town, the lake or park you're in.
- Label colours, size, glow and how smoothly they follow curves are part of each map style, and dark styles get light lettering.
- Label amount in each style: Density sets how far down the list labels go, from towns only to every street and mark, and Repeat how often a road's name comes back along it. Labels start smaller than before, so more of them fit.

### Directions
- Navigate by car, by bike or on foot: turn one on in the panel, then tap where you want to go.
- Times on foot and by bike come from the actual slope of the ground along the way. Turn-by-turn names the streets and trails.
- Bridges and tunnels are handled properly: a route never "turns" onto the road under an overpass.
- Works in the 1 m areas for now.

### Recording a track
- Record track keeps going with the screen off, and is gentle on the battery: the GPS chip collects points on its own and hands them over every minute and a half, and when you stop moving the GPS rests until you move again.
- Tracks keep exactly what the GPS measured, including its height.

### Height
- The GPS's own height above sea level, next to the map's elevation where you are. It uses the shape of the Earth's gravity (the geoid), now part of the map.

### Keeping current
- Mahere updates itself: Settings → Updates shows the release and development builds; release updates install automatically on Wi-Fi.
- Places you've already viewed refresh themselves when the map is improved.
- A Linux version, alongside Android.

### New detail
- Seattle and Puget Sound at 1 m, with the seabed under the Sound.
- Northwest Montana at 1 m.
- Flathead Lake and other big lakes fixed in the 1 m areas, and bad elevation spots at the edges of the lidar removed.

### Fixes
- The app could freeze over places with little detail; it no longer can.
- The GPS switches off whenever the app isn't in front, unless a track is recording.
- The pen choice and colour are remembered, and a stylus always draws.
