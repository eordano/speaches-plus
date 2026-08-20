import os
from PIL import Image, ImageDraw, ImageFont

FONTS = os.environ.get("FONTS", "/usr/share/fonts/truetype/dejavu")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rendered"))

def F(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)

sans = lambda s: F("DejaVuSans.ttf", s)
sansb = lambda s: F("DejaVuSans-Bold.ttf", s)
serif = lambda s: F("DejaVuSerif.ttf", s)
mono = lambda s: F("DejaVuSansMono.ttf", s)

def new_page(w=1400, h=1900):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    return img, ImageDraw.Draw(img)

def save(img, lines, name):
    os.makedirs(OUT, exist_ok=True)
    img.save(os.path.join(OUT, name + ".png"))
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)

def page1():
    img, d = new_page()
    gt = []
    def t(x, y, s, font):
        d.text((x, y), s, fill=(0, 0, 0), font=font)
        gt.append(s)
    t(80, 70, "NORTHWIND SUPPLY CO.", sansb(44))
    t(80, 130, "1454 Harbor Street, Portsmouth, NH 03801", sans(26))
    t(80, 165, "billing@northwindsupply.example", sans(26))
    t(950, 80, "INVOICE #2041", sansb(32))
    t(950, 125, "Date: 2026-07-18", sans(26))
    t(950, 160, "Due: 2026-08-17", sans(26))
    t(80, 260, "Bill To: Meridian Labs, 22 Quay Road, Boston, MA 02110", sans(28))
    y = 380
    rows = [
        ("Item", "Qty", "Unit Price", "Amount"),
        ("Steel brackets (large)", "40", "3.25", "130.00"),
        ("Hex bolts M8", "500", "0.12", "60.00"),
        ("Powder coating service", "1", "220.00", "220.00"),
        ("Delivery surcharge", "1", "35.50", "35.50"),
    ]
    for i, (a, b, c, e) in enumerate(rows):
        f = sansb(28) if i == 0 else sans(28)
        d.text((100, y), a, fill=(0, 0, 0), font=f)
        d.text((640, y), b, fill=(0, 0, 0), font=f)
        d.text((820, y), c, fill=(0, 0, 0), font=f)
        d.text((1080, y), e, fill=(0, 0, 0), font=f)
        gt.append(f"{a} {b} {c} {e}")
        y += 52
        if i == 0:
            d.line((90, y - 8, 1300, y - 8), fill=(0, 0, 0), width=2)
    d.line((90, y + 4, 1300, y + 4), fill=(0, 0, 0), width=2)
    t(820, y + 30, "Subtotal: 445.50", sans(28))
    t(820, y + 75, "Tax (8%): 35.64", sans(28))
    t(820, y + 120, "Total: 481.14", sansb(32))
    t(80, y + 240, "Payment is due within 30 days. Late payments accrue 1.5% monthly interest.", sans(26))
    t(80, y + 280, "Please reference invoice number 2041 on all correspondence.", sans(26))
    save(img, gt, "real-01-invoice")

def page2():
    img, d = new_page()
    gt = []
    def t(x, y, s, font):
        d.text((x, y), s, fill=(0, 0, 0), font=font)
        gt.append(s)
    t(80, 60, "THE COASTAL OBSERVER", serif(46))
    t(80, 130, "Vol. XII, No. 34 - Saturday Edition", serif(24))
    d.line((80, 180, 1320, 180), fill=(0, 0, 0), width=3)
    t(80, 210, "Harbor Renovation Enters Final Phase", sansb(34))
    col1 = [
        "The decade-long effort to restore the",
        "eastern breakwater reached a milestone",
        "this week as engineers poured the last",
        "of the reinforced concrete caissons.",
        "City officials expect the harbor to",
        "reopen to commercial traffic by early",
        "October, weather permitting.",
        "Local fishermen, who have rerouted",
        "through the narrow channel since 2019,",
        "welcomed the news with cautious",
        "optimism.",
    ]
    col2 = [
        "Funding for the project came from a",
        "mix of federal grants and municipal",
        "bonds totaling 48 million dollars.",
        "An oversight committee will publish",
        "its final audit in November.",
        "Meanwhile, the marina association is",
        "planning a reopening festival with",
        "boat parades, food stalls, and a",
        "lantern ceremony at dusk.",
        "Organizers expect several thousand",
        "visitors over the weekend.",
    ]
    y = 290
    for s in col1:
        d.text((80, y), s, fill=(0, 0, 0), font=serif(27))
        gt.append(s)
        y += 42
    y = 290
    for s in col2:
        d.text((720, y), s, fill=(0, 0, 0), font=serif(27))
        gt.append(s)
        y += 42
    d.line((80, 800, 1320, 800), fill=(0, 0, 0), width=2)
    t(80, 830, "Tide Tables, Page 7 | Classifieds, Page 9 | Weather: NE winds 10-15 kt", sans(24))
    save(img, gt, "real-02-newspaper")

def page3():
    img, d = new_page()
    gt = []
    def t(x, y, s, font):
        d.text((x, y), s, fill=(0, 0, 0), font=font)
        gt.append(s)
    t(80, 70, "Quarterly Engineering Report", sansb(40))
    t(80, 140, "Q2 2026 - Platform Infrastructure Team", sans(28))
    t(80, 240, "1. Summary", sansb(32))
    t(80, 295, "Service availability held at 99.97% across all regions this quarter.", sans(27))
    t(80, 335, "The migration to the new scheduler completed two weeks ahead of plan.", sans(27))
    t(80, 425, "2. Key Metrics", sansb(32))
    for i, s in enumerate([
        "- Median API latency: 41 ms (down from 58 ms)",
        "- Error budget consumed: 22 percent",
        "- Deployments per week: 134",
        "- Mean time to recovery: 11 minutes",
    ]):
        t(110, 480 + i * 44, s, sans(27))
    t(80, 700, "3. Incidents", sansb(32))
    t(80, 755, "Two SEV-2 incidents occurred in May, both traced to a faulty rollout of", sans(27))
    t(80, 795, "the connection pooler. A full postmortem is linked in the appendix.", sans(27))
    t(80, 885, "4. Next Quarter", sansb(32))
    t(80, 940, "We will pilot zonal failover drills and finish the IPv6 rollout.", sans(27))
    t(80, 1030, "Prepared by: R. Alvarez, approved by the infrastructure steering group.", sans(24))
    save(img, gt, "real-03-report")

def page4():
    img, d = new_page()
    gt = []
    def t(x, y, s, font):
        d.text((x, y), s, fill=(0, 0, 0), font=font)
        gt.append(s)
    t(900, 80, "14 Elm Street", serif(28))
    t(900, 120, "Concord, MA 01742", serif(28))
    t(900, 160, "July 30, 2026", serif(28))
    t(80, 280, "Dear Dr. Whitfield,", serif(30))
    body = [
        "Thank you for hosting our research group last Thursday. The tour of the",
        "annex facility was illuminating, and the students have not stopped talking",
        "about the tidal simulation tank.",
        "We would like to schedule a follow-up visit in September to collect water",
        "samples for the comparative salinity study. Our equipment list is enclosed,",
        "along with the proposed sampling schedule.",
        "Please let me know whether the week of September 14 works for your team.",
    ]
    y = 360
    for s in body:
        t(120, y, s, serif(28))
        y += 46
    t(120, y + 60, "With gratitude,", serif(30))
    t(120, y + 140, "Prof. Elena Marsh", serif(30))
    t(120, y + 185, "Department of Oceanography, Bayview College", serif(26))
    save(img, gt, "real-04-letter")

def page5():
    img, d = new_page()
    gt = []
    def t(x, y, s, font):
        d.text((x, y), s, fill=(0, 0, 0), font=font)
        gt.append(s)
    t(80, 60, "LAB NOTEBOOK - BATCH 47", mono(36))
    t(80, 140, "Objective: verify the catalyst loading sweep from 0.5% to 4.0%.", sans(28))
    y = 230
    rows = [
        ("Run", "Loading", "Yield", "Purity"),
        ("47-A", "0.5%", "61.2%", "97.1%"),
        ("47-B", "1.0%", "74.8%", "96.8%"),
        ("47-C", "2.0%", "88.3%", "96.5%"),
        ("47-D", "4.0%", "87.9%", "94.2%"),
    ]
    for i, r in enumerate(rows):
        f = mono(30) if i == 0 else mono(28)
        for j, cell in enumerate(r):
            d.text((100 + j * 280, y), cell, fill=(0, 0, 0), font=f)
        gt.append(" ".join(r))
        y += 50
    t(80, y + 40, "Observation: yield plateaus above 2.0% loading while purity declines.", sans(28))
    t(80, y + 90, "Next step: repeat 47-C in triplicate and submit samples for NMR.", sans(28))
    t(80, y + 190, "Signed: J. Okafor    Witnessed: T. Lindqvist    Date: 2026-08-02", sans(26))
    save(img, gt, "real-05-labnotes")

page1()
page2()
page3()
page4()
page5()
