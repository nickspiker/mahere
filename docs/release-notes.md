# Mahere release notes

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
