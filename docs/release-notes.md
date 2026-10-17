# Mahere release notes

## 0.21x · October 2026

### On your computer
- Mahere for Mac (Apple Silicon) and Windows, beside Linux, updating itself like the phone.

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
