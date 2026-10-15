# Label fonts

Typefaces for map lettering (Nick 2026-10-14: a few base fonts, ASCII only, so a theme can set its own type).
Each is subset to printable ASCII plus the middle dot, degree, multiplication sign and right quote, hinting dropped; letters outside fall back to the panel's Open Sans.
The variable fonts keep their weight axis (wght), and fluor on cosmic-text 0.19 draws them at any weight; a width axis is pinned at 100.
Cut with fontTools from the google/fonts repository, 2026-10-14.

| Family here | File | From | Licence |
|---|---|---|---|
| Oxanium | Oxanium.ttf, wght 200-800 | Oxanium (variable) | OFL-oxanium.txt |
| EB Garamond | EBGaramond.ttf, wght 400-800 | EB Garamond (variable) | OFL-ebgaramond.txt |
| Roboto Condensed | RobotoCondensed.ttf, wght 100-900 | Roboto Condensed (variable) | OFL-robotocondensed.txt |
| Fredoka | Fredoka.ttf, wght 300-700 | Fredoka (variable, width pinned at 100) | OFL-fredoka.txt |
| Cinzel | Cinzel.ttf, wght 400-900 | Cinzel (variable) | OFL-cinzel.txt |
| Open Sans Variable | OpenSansVariable.ttf, wght 300-800 | Open Sans (variable, width pinned at 100), named apart from the panel's static Open Sans so it is the one matched at every weight | OFL-opensans.txt |
| Mahere Mono | MahereMono-300 to -700.ttf, five static weights | IBM Plex Mono (no variable master), renamed as the licence's Reserved Font Name "Plex" requires of a modified version | OFL-ibmplexmono.txt |

These are Modified Versions under the SIL Open Font License 1.1 and are distributed under it.
