import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

from fontfind import font_file

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures-scale"
)

FONT_REG = font_file("LiberationSans-Regular.ttf", "FONT_REG")
FONT_BOLD = font_file("LiberationSans-Bold.ttf", "FONT_BOLD")
FONT_MONO = font_file("LiberationMono-Regular.ttf", "FONT_MONO")
GENERATOR = "generate-090-scale-fixtures.py (PIL, %s + %s)" % (
    os.path.basename(FONT_REG), os.path.basename(FONT_MONO))

CAP_MAX_PIXELS = 11_289_600


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(prose, width):
    words = prose.split()
    lines, cur = [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


class Page:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.img = Image.new("RGB", (w, h), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.regions = []
        self.lines = []

    def block(self, x, y, lines, size, tier, bold=False, mono=False, gap=0.30,
              fill=(0, 0, 0)):
        path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
        f = font(path, size)
        x0, y0, x1, y1 = math.inf, math.inf, 0, 0
        cy = y
        for line in lines:
            bb = self.d.textbbox((x, cy), line, font=f)
            if bb[2] > self.w - 20:
                raise SystemExit(f"overflow: {line!r} size={size} ends at {bb[2]} > {self.w}")
            self.d.text((x, cy), line, fill=fill, font=f)
            x0, y0 = min(x0, bb[0]), min(y0, bb[1])
            x1, y1 = max(x1, bb[2]), max(y1, bb[3])
            self.lines.append(line)
            cy += int(size * (1 + gap))
        self.regions.append({
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "tier": tier,
            "text": "\n".join(lines),
        })
        return cy

    def save(self, name, family, description, extra=None):
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        self.img.save(os.path.join(d, "input.png"), optimize=True)
        mp = self.w * self.h / 1e6
        downscale = min(1.0, math.sqrt(CAP_MAX_PIXELS / (self.w * self.h)))
        fx = {
            "name": name,
            "family": family,
            "description": description,
            "input": "input.png",
            "expected_text": "\n".join(self.lines),
            "render": {
                "generator": GENERATOR,
                "page_px": [self.w, self.h],
                "megapixels": round(mp, 2),
                "single_pass_downscale_at_default_cap": round(downscale, 3),
            },
            "regions": self.regions,
            "comparison_strategy": "cer",
            "skip_when_no_model": True,
        }
        if extra:
            fx.update(extra)
        with open(os.path.join(d, "fixture.json"), "w") as fh:
            json.dump(fx, fh, indent=2)
        px = os.path.getsize(os.path.join(d, "input.png"))
        print(f"wrote {name}: {self.w}x{self.h} ({mp:.1f}MP) png={px/1e6:.1f}MB "
              f"regions={len(self.regions)} downscale={downscale:.3f}")


def catalog_15mp():
    p = Page(3248, 4592)
    m = 140
    bw = 120
    y = p.block(m, 180, ["HARBORLIGHT INSTRUMENTS"], 96, "heading", bold=True)
    y = p.block(m, y + 60, ["Precision Components Catalog, Edition 14"], 64, "heading")
    y = p.block(m, y + 130, ["Section 3: Optical Assemblies"] + wrap(
        "All housings are machined from 6061-T6 aluminium and anodized in matte "
        "black. Threads follow the RMS standard unless noted otherwise in the "
        "item description. Every assembly ships with a test card recording the "
        "measured transmitted wavefront and the coating run code.", bw,
    ), 42, "body")
    y = p.block(m, y + 110, [
        "OA-2210 Collimator, 25 mm aperture, serial range LN-4821 to LN-5140",
        "OA-2214 Beam expander 3x, fused silica, coated 400-700 nm, code BX-31",
        "OA-2290 Kinematic mount, 80 TPI adjusters, lockable, code KM-80",
    ], 42, "body")
    y = p.block(m, y + 140, ["Ordering and Lead Times"], 64, "heading", bold=True)
    y = p.block(m, y + 60, wrap(
        "Orders received before 14:00 UTC ship the same business day from the "
        "Kiel works. Custom coatings add nine working days to any line item. "
        "Quote code QX-77 applies to universities and public research "
        "laboratories only and cannot combine with distributor pricing.", bw,
    ), 42, "body")
    y = p.block(m, y + 140, [
        "Batch acceptance: MIL-PRF-13830B, scratch-dig 40-20 on all catalogue optics.",
        "Interferometric flatness lambda/10 at 632.8 nm available on request per lot.",
    ], 26, "fine")
    y = p.block(m, y + 90, [
        "Harborlight Instruments GmbH, Werftstrasse 21, 24103 Kiel. Registered HRB 8821 KI.",
        "VAT DE 318 442 901. Catalog data sheet HL-CAT-14-3EN. Errata via the service desk.",
    ], 26, "fine")
    y = p.block(m, y + 160, ["Section 4: Mounting Hardware"], 64, "heading", bold=True)
    y = p.block(m, y + 60, [
        "MH-410 Post holder, 40 mm bore, side-locking thumbscrew, black anodize",
        "MH-415 Damped post, filled core, first resonance above 900 Hz measured",
        "MH-460 Rail carrier, dovetail profile, 25 mm of travel per adjuster turn",
    ], 42, "body")
    y = p.block(m, y + 140, ["Warranty and Returns"], 64, "heading", bold=True)
    y = p.block(m, y + 60, wrap(
        "All items carry a three year warranty against defects in materials and "
        "workmanship. Return authorisations issue from service desk RMA-9 within "
        "one business day of request, and advance replacements are available for "
        "any item stocked in the Kiel or Portsmouth warehouses.", bw,
    ), 42, "body")
    y = p.block(m, y + 140, [
        "Prices exclude carriage and import duty. INCOTERMS 2020 EXW Kiel applies throughout.",
        "This edition supersedes HL-CAT-13 in whole. The errata list is available on request.",
    ], 26, "fine")
    y = p.block(m, y + 160, ["Section 5: Cleaning and Handling"] + wrap(
        "Handle coated optics by their edges with powder-free gloves and blow "
        "loose dust away with dry nitrogen before any contact method. Use "
        "spectroscopic grade acetone on lint-free wipes, one pass per wipe "
        "surface, and never rewet a used wipe. Store optics in their shipped "
        "membrane boxes below 60 percent relative humidity.", bw,
    ), 42, "body")
    p.block(m, y + 140, ["Technical Support"] + wrap(
        "Application engineers answer at support@harborlight.example between "
        "07:00 and 18:00 CET on working days. Include the serial number and the "
        "coating run code from the edge engraving with any query. Calibration "
        "certificates reissue free within three years of shipment, and "
        "out-of-warranty repairs are quoted within five working days.", bw,
    ), 42, "body")
    p.save(
        "090-scale-15mp-catalog", "090-scale",
        "15MP 300dpi-class catalog page: the tier-policy boundary fixture; "
        "single-pass downscale 0.87x keeps every glyph tier legible.",
    )


def report_35mp():
    p = Page(4956, 7016)
    m = 220
    y = p.block(m, 260, ["GEOTHERMAL TEST WELL KV-9"], 160, "heading", bold=True)
    y = p.block(m, y + 90, ["Quarterly Integrity Report, Q3 2026"], 96, "heading")
    y = p.block(m, y + 200, [
        "1. Summary of Operations",
        "The wellhead held a stable 41.3 bar across the quarter. Flow",
        "testing on 12 August reached 96 kg/s at 243 degrees Celsius,",
        "matching the drawdown model within 2.1 percent.",
    ], 84, "body")
    y = p.block(m, y + 220, [
        "2. Casing Inspection",
        "Ultrasonic logs show wall thickness above 11.2 mm everywhere",
        "except joint 41, where pitting reduced it to 9.7 mm. The joint",
        "remains inside the 8.5 mm retirement threshold.",
    ], 84, "body")
    y = p.block(m, y + 220, [
        "3. Instrumentation",
        "Downhole gauge DG-118 drifted 0.4 bar and was recalibrated.",
        "Fibre line FL-2 returned clean DTS traces after splice repair.",
    ], 84, "body")
    y = p.block(m, y + 260, ["Appendix A: Sample Assays"], 96, "heading", bold=True)
    y = p.block(m, y + 100, [
        "Sample KV9-S31: silica 612 mg/kg, chloride 88 mg/kg, pH 7.9",
        "Sample KV9-S32: silica 598 mg/kg, chloride 91 mg/kg, pH 8.0",
        "Sample KV9-S33: silica 640 mg/kg, chloride 84 mg/kg, pH 7.8",
    ], 50, "fine", mono=True)
    ycols = y + 200
    yl = p.block(m, ycols, [
        "Note 1: Assay uncertainty is 3 percent",
        "relative, one sigma, per lab method",
        "LM-410 revision D.",
    ], 33, "micro")
    yr = p.block(2700, ycols, [
        "Note 2: Chloride trend remains below",
        "the scaling advisory limit defined in",
        "operations memo OM-2214.",
    ], 33, "micro")
    y = p.block(m, max(yl, yr) + 200, [
        "4. Reinjection Performance",
        "Injection well KV-9R accepted the full 71 kg/s return stream at",
        "18.9 bar wellhead pressure. Tracer test T-14 shows 4.1 percent",
        "breakthrough after 62 days, unchanged from the spring campaign.",
    ], 84, "body")
    y = p.block(m, y + 220, [
        "5. Planned Work",
        "A caliper rerun of joints 38 through 44 is scheduled for early",
        "November, together with replacement of the annulus bleed valve",
        "and a full recalibration of the fiscal metering skid.",
    ], 84, "body")
    y = p.block(m, y + 260, ["Appendix B: Vibration Survey"], 96, "heading", bold=True)
    y = p.block(m, y + 100, [
        "Pump P-201: 2.1 mm/s RMS at bearing housing, within ISO zone B",
        "Pump P-202: 4.8 mm/s RMS, rising trend, realignment scheduled",
        "Separator feed line: no cavitation signature above 2 kilohertz",
    ], 50, "fine", mono=True)
    y = p.block(m, y + 200, [
        "6. Environmental Compliance",
        "Hydrogen sulphide at the site boundary averaged 3.1 micrograms",
        "per cubic metre, well under the 15 microgram consent ceiling.",
        "Brine pond levels stayed within the freeboard requirement and",
        "no discharge events were recorded during the quarter.",
    ], 84, "body")
    p.block(m, y + 180, [
        "Report KV9-Q3-2026-INT, revision 2. Approved by R. Steinsdottir, 30 September 2026.",
    ], 50, "fine")
    p.save(
        "090-scale-35mp-report", "090-scale",
        "35MP A4-at-600dpi report: fine (50px) and micro (33px) tiers sit at "
        "0.57x single-pass downscale; the micro notes section is two-column.",
    )


def broadsheet_50mp():
    p = Page(5936, 8408)
    m = 260
    y = p.block(m, 300, ["THE MERIDIAN LEDGER"], 180, "heading", bold=True)
    y = p.block(m, y + 100, ["Founded 1908. Weekend Edition, 14 November 2026."], 90, "heading")
    y = p.block(m, y + 240, ["Tidal Array Delivers First Power to Grid"] + wrap(
        "The Stormness tidal array fed 18 megawatts into the northern grid on "
        "Thursday, three weeks ahead of schedule. Engineers credit the "
        "redesigned blade pitch controller, which survived the spring tides "
        "that destroyed the 2023 prototype field, and a subsea hub that lets "
        "divers isolate one turbine without darkening the whole string.", 105,
    ), 90, "body")
    y = p.block(m, y + 260, wrap(
        "Turbine seven remains offline after a seal failure flooded its nacelle "
        "in October. Divers recovered the unit intact and the operator expects "
        "a rebuilt gearbox to spin by January, with the insurance survey "
        "already closed out and spares landed at the quayside store.", 105,
    ), 90, "body")
    y = p.block(m, y + 300, ["Shipping Notices"], 110, "heading", bold=True)
    y = p.block(m, y + 110, [
        "MV Kittiwake, berth 4, discharging aggregate until Tuesday.",
        "MV Solan Goose, berth 9, loading 2,400 t of cured timber.",
        "Pilot boarding moves to the eastern anchorage from 06:00.",
    ], 50, "fine")
    y = p.block(m, y + 240, [
        "Tide table, Stormness Sound: high 04:41 and 17:02, low 10:48 and 23:11.",
        "Heights 3.4 m and 3.1 m. Springs return on the 21st. Chart datum LAT.",
        "Notice 2026-118: racon on the Skerry buoy unreliable until further notice.",
    ], 33, "micro")
    y = p.block(m, y + 200, [
        "Classified index: employment 12, marine sales 14, legal notices 15, tenders 17.",
        "Printed at Quayside Works on 45 gsm newsprint. ISSN 0140-7723. Price two pounds.",
    ], 33, "micro")
    y = p.block(m, y + 300, ["Council Approves Breakwater Levy"], 110, "heading", bold=True)
    y = p.block(m, y + 120, wrap(
        "A harbour levy of 40 pence per landed tonne passed by nine votes to "
        "three on Wednesday night. Proceeds are ring-fenced for the eastern "
        "breakwater rebuild, estimated at 6.2 million pounds over four seasons "
        "of work, and the first caisson contract goes to tender in February.", 105,
    ), 90, "body")
    y = p.block(m, y + 260, wrap(
        "Fisheries representatives abstained, calling instead for a berthing "
        "rebate for vessels under twelve metres. The council agreed to review "
        "the rebate at the March sitting once a season of levy receipts can be "
        "set against the dredging account.", 105,
    ), 90, "body")
    y = p.block(m, y + 300, ["Weather and Sea State"], 110, "heading", bold=True)
    y = p.block(m, y + 110, [
        "Saturday: southwest gale eight, easing six by evening. Rain.",
        "Sunday: west five, showers, good visibility between squalls.",
        "Sea area Fair Isle: rough becoming moderate. Icing nil.",
    ], 50, "fine")
    y = p.block(m, y + 220, [
        "Corrections: last week's edition gave the Kittiwake's tonnage as 4,100; the correct",
        "figure is 4,010 gross tonnes. The Ledger regrets the error. Contact corrections@ledger.",
    ], 33, "micro")
    y = p.block(m, y + 280, ["Ferry Timetable Consultation Opens"] + wrap(
        "The winter timetable consultation runs until 5 December. Proposals cut "
        "the Tuesday freight-only sailing and add a late Friday passenger run. "
        "Responses go to the transport partnership office or any harbour "
        "master's post box before the closing date.", 105,
    ), 90, "body")
    p.block(m, y + 240, ["School Boat Prize Returns to Northside"] + wrap(
        "Northside Secondary took the inter-school sailing trophy for the third "
        "year, finishing four points clear over the combined southern crews in "
        "a shifting easterly breeze that favoured the lighter boats.", 105,
    ), 90, "body")
    p.save(
        "090-scale-50mp-broadsheet", "090-scale",
        "50MP broadsheet scan: 0.475x single-pass downscale takes micro print "
        "to a 16px em; the acceptance fixture the tiered path must read.",
    )


def archive_100mp():
    p = Page(8400, 11888)
    m = 380
    y = p.block(m, 420, ["NATIONAL SURVEY ARCHIVE"], 220, "heading", bold=True)
    y = p.block(m, y + 130, ["Plate 47: Coastal Triangulation, 1911 Resurvey"], 120, "heading")
    y = p.block(m, y + 300, wrap(
        "Digitised at 1200 dpi from the original vellum plate. The station "
        "coordinates below are transcribed from the field ledger and keep the "
        "original Clarke 1880 figure, so users comparing against the modern "
        "network must apply the published datum shift before any computation.", 135,
    ), 100, "body")
    y = p.block(m, y + 280, [
        "Station BEACON HILL: 58 deg 12 min 41.30 sec N, 6 deg 22 min 08.71 sec W",
        "Station CNOC ARD:    58 deg 11 min 57.02 sec N, 6 deg 19 min 44.55 sec W",
        "Station SGEIR DUBH:  58 deg 13 min 22.87 sec N, 6 deg 17 min 30.12 sec W",
    ], 84, "body", mono=True)
    y = p.block(m, y + 320, ["Observer Notes"], 120, "heading", bold=True)
    y = p.block(m, y + 130, wrap(
        "Heliograph signals from Beacon Hill were lost in haar on the morning "
        "rounds and all angles were repeated after noon. The theodolite, "
        "Troughton and Simms No. 3, read to one second of arc, and each round "
        "of angles closed the horizon within three seconds before acceptance.", 135,
    ), 100, "body")
    y = p.block(m, y + 300, [
        "Reduction follows Ordnance memorandum 44 of 1909. Spherical excess",
        "distributed equally. Closing error of triangle 47-C: 1.88 seconds.",
    ], 50, "fine")
    y = p.block(m, y + 220, [
        "Conservation record: plate cleaned and rebacked March 1961; fungal foxing at NE corner",
        "stabilised. Access copy ASA-47-1911-R2. Original held in vault 9, humidity class B.",
    ], 33, "micro")
    y = p.block(m, y + 360, ["Baseline Measurements"], 120, "heading", bold=True)
    y = p.block(m, y + 140, [
        "The Loch Roag base of 1909 was remeasured with invar tapes",
        "numbers 61 and 63 under 15 pounds tension. Twelve bays gave",
        "a mean of 3,662.114 metres with a spread of 4.1 millimetres.",
    ], 100, "body")
    y = p.block(m, y + 300, [
        "Tape 61 standardisation: 30.000,42 m at 14.2 C, coefficient 0.0000006",
        "Tape 63 standardisation: 30.000,38 m at 14.2 C, coefficient 0.0000007",
        "Field temperatures by mercury thermometers 118 and 121, read to 0.2 C",
    ], 84, "body", mono=True)
    y = p.block(m, y + 340, ["Adjustment Summary"], 120, "heading", bold=True)
    y = p.block(m, y + 140, [
        "Least squares adjustment over 31 conditions closed the network",
        "with a maximum residual of 0.94 seconds on line 47-52. Side",
        "lengths are tabulated on plate 48 with their logarithms.",
    ], 100, "body")
    y = p.block(m, y + 300, [
        "Archivist note: the 1911 party lists J. Macleod as observer and A. Ross as recorder.",
        "Wages ledger cross-reference WL-1911-224. Photograph album PA-19 shows camp at Cnoc Ard.",
    ], 50, "fine")
    y = p.block(m, y + 240, [
        "Digitisation metadata: scanner SG-1200-IV, target IT8.7/2 batch 2026-03, delta E mean 1.1.",
        "File checksum sha256 9c41...e7 recorded in accession register AR-2026-118. Scanned by K.O.",
    ], 33, "micro")
    y = p.block(m, y + 320, ["Comparison with the 1852 Primary Triangulation"] + wrap(
        "Where the 1911 stations reoccupy primary marks, coordinate shifts stay "
        "below 0.4 metres except at Sgeir Dubh, where the cairn had been "
        "rebuilt seaward of the original centre mark. A buried bottle holding "
        "the 1852 centre punch was recovered at half a metre and the new "
        "pillar was set over it directly, restoring continuity of the mark.", 135,
    ), 100, "body")
    y = p.block(m, y + 320, ["Gravity and Deflection Observations"] + wrap(
        "Pendulum gravity at Beacon Hill agrees with the Bouguer prediction "
        "within 11 milligals. Astronomical latitude shows a deflection of the "
        "vertical of 4.2 seconds south, ascribed to the mass of the gneiss "
        "ridge east of the station, and the azimuth condition was therefore "
        "carried on Cnoc Ard instead.", 135,
    ), 100, "body")
    y = p.block(m, y + 320, ["Availability and Citation"] + wrap(
        "High resolution scans of plates 44 through 52 are available on reader "
        "request under licence NLA-OG-4. Cite this plate as National Survey "
        "Archive, Coastal Triangulation Resurvey of 1911, plate 47, accession "
        "AR-2026-118, digitised 2026. Derived vector data carries the same "
        "licence and must reproduce this citation in full.", 135,
    ), 100, "body")
    y = p.block(m, y + 320, ["Related Holdings"] + wrap(
        "The archive holds the computation books for divisions 44 to 52, the "
        "observer's diary for the 1911 season, and glass negatives of the "
        "station cairns. Requests for the diary require a conservation review "
        "because the binding is friable along the spine.", 135,
    ), 100, "body")
    p.block(m, y + 240, wrap(
        "Reader services: the search room opens Tuesday to Saturday, 09:30 to "
        "16:45, with document ordering closing at 15:30. Photography is "
        "permitted without flash under permit PH-11, and bulk imaging requests "
        "route through the digitisation programme office.", 135,
    ), 100, "body")
    p.save(
        "090-scale-100mp-archive", "090-scale",
        "100MP stretch fixture: 0.336x single-pass downscale kills even the "
        "50px fine tier; segmentation is the only readable path.",
    )


def planroom_50mp():
    p = Page(8426, 5960)
    d = p.d
    grey = (176, 176, 176)
    d.rectangle((60, 60, 8366, 5900), outline=grey, width=3)
    for gx in range(600, 4200, 450):
        d.line((gx, 2700, gx, 5700), fill=grey, width=2)
    for gy in range(2700, 5701, 500):
        d.line((600, gy, 4150, gy), fill=grey, width=2)
    d.rectangle((900, 3100, 2400, 4300), outline=grey, width=6)
    d.rectangle((2700, 3600, 3900, 5300), outline=grey, width=6)
    d.ellipse((4600, 3200, 5600, 4200), outline=grey, width=5)
    d.line((4600, 5000, 8200, 5000), fill=grey, width=8)
    lx, rx = 220, 4460
    y = p.block(lx, 200, ["SITE PLAN SP-04: QUAYSIDE ENERGY CENTRE"], 64, "heading", bold=True)
    y = p.block(lx, y + 60, ["Scale 1:200 at A0. Grid: local datum QEC-2026. Sheet 4 of 12."],
                24, "body")
    y = p.block(lx, y + 90, ["GENERAL NOTES"] + wrap(
        "1. All dimensions are in millimetres unless noted. Verify every setting-out "
        "dimension on site against the QEC-2026 grid monuments before excavation. Report "
        "discrepancies over 15 mm to the engineer before proceeding with the affected work.",
        95,
    ), 20, "body")
    y = p.block(lx, y + 70, wrap(
        "2. Underground HV cable route 11kV-R2 crosses the northern access at chainage "
        "48.20. Hand dig within 1500 mm of the marked route. Cable cover tiles must be "
        "reinstated with warning tape at 300 mm above the duct crown.", 95,
    ), 20, "body")
    y = p.block(lx, y + 70, wrap(
        "3. Drainage runs DR-7 to DR-11 fall at 1:150 toward manhole MH-19. Invert "
        "levels are tabulated on sheet SP-09. Flexible couplings within 600 mm of every "
        "structure face, both sides.", 95,
    ), 20, "body")
    y = p.block(lx, y + 70, wrap(
        "4. The attenuation tank base slab is cast on 50 mm blinding over compacted "
        "type 1 fill at 95 percent modified proctor. Waterstop at every construction "
        "joint; the kicker pour includes the north wall starter bars.", 95,
    ), 20, "body")
    y = p.block(lx, y + 90, ["SETTING OUT TABLE"] + [
        "Point ST-01: E 4821.440  N 9310.220  cover level 5.840",
        "Point ST-02: E 4874.905  N 9310.220  cover level 5.815",
        "Point ST-03: E 4874.905  N 9268.130  cover level 5.760",
        "Point ST-04: E 4821.440  N 9268.130  cover level 5.790",
        "Benchmark TBM-2: brass stud in fuel store plinth, level 6.112",
    ], 18, "fine", mono=True)
    y = p.block(lx, y + 90, ["EARTHWORKS"] + wrap(
        "Strip topsoil to 300 mm and stockpile on the western bund at a maximum height "
        "of 2.5 metres. Cut material classified 2C may be reused in landscape mounding "
        "north of gridline 14 subject to moisture conditioning between 12 and 16 percent.",
        95,
    ), 20, "body")
    p.block(lx, y + 70, wrap(
        "Excavations deeper than 1200 mm require trench support to the temporary works "
        "designer's details. Groundwater strike below 4.1 metres was recorded in "
        "borehole BH-06; allow for sump pumping in the tank excavation.", 95,
    ), 20, "body")
    ry = p.block(rx, 200, ["DOOR AND GATE SCHEDULE"] + [
        "G-01 sliding gate  6000 x 2400  powder RAL 7016  motor DK-9",
        "G-02 swing gate    1200 x 2400  powder RAL 7016  keyed alike",
        "D-01 fuel store    1000 x 2100  steel FD60       panic bar",
        "D-02 switchroom     900 x 2100  steel FD60       louvre 600",
        "D-03 control room  1000 x 2100  timber FD30S     vision panel",
    ], 18, "fine", mono=True)
    ry = p.block(rx, ry + 90, ["FENCING"] + wrap(
        "Perimeter fence type F2: 2.4 metre paladin mesh on 60 x 60 SHS posts at 3.0 "
        "metre centres, set 750 mm into C20 concrete. Three lines of barbed tape above "
        "the switchyard section only, per the security consultant's memo SC-11.", 95,
    ), 20, "body")
    ry = p.block(rx, ry + 70, wrap(
        "Acoustic barrier along gridline 2 from chainage 0 to 64: 3.6 metre absorptive "
        "panels, minimum surface density 20 kilograms per square metre, colour to match "
        "RAL 7035. Posts designed for a 3 second gust of 24 metres per second.", 95,
    ), 20, "body")
    ry = p.block(rx, ry + 90, ["SERVICES CROSSING NOTES"] + wrap(
        "Telecoms duct T-4 enters the site under the eastern verge at 900 mm depth and "
        "rises into cabinet CAB-2. Maintain 300 mm clearance from the LV route and 500 "
        "mm from drainage crossings. Marker posts at every change of direction.", 95,
    ), 20, "body")
    ry = p.block(rx, ry + 90, ["REVISION TABLE"] + [
        "P1  2026-05-02  first issue for comment            KMB",
        "P2  2026-06-18  tank moved 1.2 m north, note 4     KMB",
        "C1  2026-07-30  issued for construction            RGA",
        "C2  2026-09-12  gate G-01 motor spec revised       RGA",
    ], 18, "fine", mono=True)
    d.rectangle((rx - 40, ry + 60, 8300, ry + 560), outline=grey, width=4)
    ry = p.block(rx, ry + 100, ["QUAYSIDE ENERGY CENTRE"], 56, "heading", bold=True)
    ry = p.block(rx, ry + 40, [
        "Drawing SP-04 revision C2. Drawn KMB, checked RGA, approved TW.",
        "Meridian Consulting Engineers, 22 Quay Road, Boston. Job 26-118.",
    ], 24, "body")
    p.save(
        "092-scale-50mp-planroom", "092-scale",
        "50MP A0-class plan sheet: 20px em notes and 18px em mono schedules at "
        "native detail; 0.475x single-pass downscale drops them under 10px em, "
        "the regime segmentation exists to save.",
    )


def gazette_24mp():
    p = Page(4200, 5712)
    m, gutter, ncols = 150, 60, 4
    colw = (4200 - 2 * m - (ncols - 1) * gutter) // ncols
    y0 = p.block(m, 160, ["THE HARBOUR GAZETTE: CLASSIFIED ADVERTISEMENTS"],
                 56, "heading", bold=True)
    top = y0 + 90
    cats = ["FOR SALE", "WANTED", "TO LET", "LOST", "FOUND", "NOTICE"]
    nouns = [
        "clinker dinghy with oars and road trailer",
        "six pane greenhouse on treated timber base",
        "cast iron peat stove with flue kit",
        "piano accordion, forty eight bass",
        "sail loft work bench with twin vices",
        "dozen crab creels, recently retagged",
        "upright spinning wheel and carders",
        "four horse outboard motor, long shaft",
        "pine dresser, two glazed doors",
        "three bee hives with spare supers",
        "galvanised feed bins, sixty gallon",
        "drystone walling hammers and pins",
        "herring net, forty fathom, good cotton",
        "pony trap with lamps and harness",
        "knitting machine with punch cards",
        "peat cutting irons, long and short",
        "ship model of the Solan Goose",
        "wind generator tower sections",
        "byre door, oak, iron hinges",
        "lobster holding box, floating type",
    ]
    n = 0
    for c in range(ncols):
        x = m + c * (colw + gutter)
        y = top
        for r in range(30):
            cat = cats[n % len(cats)]
            noun = nouns[n % len(nouns)]
            box = 101 + n
            price = 15 + 7 * n
            body = f"{cat}: {noun}. {price} pounds or near offer. Box {box}."
            y = p.block(x, y, wrap(body, 68), 24, "entry", gap=0.22)
            y += 58
            n += 1
    p.save(
        "093-scale-24mp-gazette", "093-scale",
        "24MP four-column classifieds page with 80 small entries: the T0 "
        "worst case, where model-scout cost is decode length per region.",
    )


UI_SCALE = {"1080p": 1.0, "1440p": 1.33, "4k": 2.0, "5k": 2.67, "uw49": 1.33}


def screen(name, w, h, key, dark=False):
    s = UI_SCALE[key]

    def z(px):
        return max(11, int(px * s))

    p = Page(w, h)
    d = p.d
    ink = (222, 222, 222) if dark else (0, 0, 0)
    if dark:
        d.rectangle((0, 0, w, h), fill=(24, 25, 28))
    chrome = (48, 49, 52) if dark else (236, 236, 236)
    panel = (36, 37, 40) if dark else (245, 245, 245)
    status = (52, 56, 66) if dark else (220, 224, 232)
    orig_block = p.block
    def block(*a, **kw):
        kw.setdefault("fill", ink)
        return orig_block(*a, **kw)
    p.block = block
    d.rectangle((0, 0, w, z(34)), fill=chrome)
    p.block(int(18 * s), int(7 * s),
            ["File   Edit   View   Terminal   Window   Help"], z(15), "menubar")
    d.rectangle((0, z(34), int(300 * s), h - z(28)), fill=panel)
    p.block(int(16 * s), z(34) + int(18 * s), [
        "PROJECT FILES",
        "flightdeck.rs",
        "telemetry.rs",
        "packet_codec.rs",
        "ground_station.toml",
        "README.md",
    ], z(16), "sidebar", gap=0.55)
    cx = int(300 * s) + int(40 * s)
    y = p.block(cx, z(34) + int(24 * s), ["telemetry.rs - flightdeck workspace"],
                z(22), "title", bold=True)
    y = p.block(cx, y + int(18 * s), [
        "pub fn decode_frame(buf: &[u8]) -> Result<Frame, CodecError> {",
        "    let sync = u16::from_be_bytes([buf[0], buf[1]]);",
        "    if sync != 0xEB90 { return Err(CodecError::BadSync(sync)); }",
        "    let apid = u16::from_be_bytes([buf[2], buf[3]]) & 0x07FF;",
        "    Frame::checked(apid, &buf[6..])",
        "}",
    ], z(16), "code", mono=True, gap=0.5)
    y = p.block(cx, y + int(30 * s), [
        "cargo test --package flightdeck",
        "running 42 tests ... 42 passed, 0 failed in 3.81s",
        "warning: field `dropped_frames` is never read",
    ], z(15), "terminal", mono=True, gap=0.5)
    if w / h > 3.0:
        rx = int(w * 0.62)
        p.block(rx, z(34) + int(24 * s), [
            "LIVE TELEMETRY",
            "Downlink lock: 8.4 dB margin",
            "Packets: 118,442 received, 12 gaps",
            "Battery bus: 28.1 V at 3.2 A",
            "Attitude error: 0.14 degrees RMS",
        ], z(18), "panel", gap=0.6)
    d.rectangle((0, h - z(28), w, h), fill=status)
    p.block(int(18 * s), h - z(24),
            ["main branch  |  UTF-8  |  Ln 214, Col 9  |  rust-analyzer ready"],
            z(14), "statusbar")
    theme = "dark theme" if dark else "light theme"
    p.save(name, "091-screen",
           f"{key} UI screen ({w}x{h}, {theme}): menu bar, file sidebar, code "
           "pane, terminal and status bar at native UI glyph sizes.")


def darkdoc_4mp():
    p = Page(1700, 2400)
    ink = (222, 222, 222)
    p.d.rectangle((0, 0, 1700, 2400), fill=(24, 25, 28))
    m = 120
    bw = 74
    y = p.block(m, 160, ["NIGHTSIDE OPERATIONS MANUAL"], 64, "heading",
                bold=True, fill=ink)
    y = p.block(m, y + 40, ["Section 7: Thermal Watch Procedures"], 44,
                "heading", fill=ink)
    y = p.block(m, y + 60, wrap(
        "During eclipse passes the radiator loop runs unpumped and the watch "
        "officer logs the plateau temperature of both accumulator tanks every "
        "eleven minutes. A divergence beyond four kelvin between the tanks "
        "indicates a stuck bypass valve and the recovery checklist on page "
        "ninety-one applies without further authorization.", bw), 34, "body",
        fill=ink)
    y = p.block(m, y + 70, wrap(
        "Heater bank C remains isolated while the survey instrument records. "
        "Its inrush transient couples into the magnetometer boom and has "
        "produced false substorm signatures twice this quarter, most recently "
        "on the descending node over the Weddell Sea anomaly.", bw), 34,
        "body", fill=ink)
    y = p.block(m, y + 90, [
        "Watch handover requires three items:",
        "1. Signed plateau log with divergence column initialed",
        "2. Valve position table read back against telemetry",
        "3. Heater bank C isolation breaker photographed",
    ], 34, "list", fill=ink)
    y = p.block(m, y + 90, wrap(
        "Deviations are recorded in the anomaly ledger before end of shift. "
        "The ledger is the controlling document in any disagreement between "
        "console notes and downlinked housekeeping frames.", bw), 34, "body",
        fill=ink)
    p.block(m, 2400 - 140, ["NOM-7 rev D  |  approved flight ops  |  page 88"],
            28, "footer", fill=ink)
    p.save("094-dark-4mp-report", "094-dark",
           "Dark-themed document render (1700x2400 portrait, light-on-dark "
           "prose): title, body paragraphs, list and footer -- the dark "
           "DOCUMENT geometry beside the 091 dark UI screen.")


def main():
    os.makedirs(OUT, exist_ok=True)
    catalog_15mp()
    report_35mp()
    broadsheet_50mp()
    archive_100mp()
    planroom_50mp()
    gazette_24mp()
    screen("091-screen-1080p-editor", 1920, 1080, "1080p")
    screen("091-screen-1440p-editor", 2560, 1440, "1440p")
    screen("091-screen-4k-editor", 3840, 2160, "4k")
    screen("091-screen-5k-editor", 5120, 2880, "5k")
    screen("091-screen-uw49-editor", 5120, 1440, "uw49")
    screen("091-screen-4k-darkmode", 3840, 2160, "4k", dark=True)
    darkdoc_4mp()


if __name__ == "__main__":
    main()
