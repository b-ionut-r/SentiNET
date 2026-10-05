"""Everyday English words that companies use as (the head of) their names.

A brand that is also an ordinary word ("Structure Therapeutics", "Align
Technology", "Viking", "Target") cannot be searched by its bare head without
drowning in unrelated text, so `app.resolve.names` keeps such names whole
("Structure Therapeutics", not "Structure") and GDELT anchors them to company
phrases. The set is hand-curated for company-name heads: nature, animals,
myth, places, abstract virtues, shapes, materials, verbs of motion and growth,
plus the homonym brands every finance tool trips over. Coined brands
("Palantir", "Moderna") and surnames ("Hilton") are deliberately absent.
"""
from __future__ import annotations

_RAW = """
apple target meta block snap visa shell amazon oracle ford gap square delta united american southwest
alphabet zoom unity lucid ball arch carnival booking progressive general discover match global snowflake
affirm upstart root lemonade oscar plug ally citizens regions marathon pioneer continental spirit frontier
monster celsius constellation crocs academy express guess tapestry columbia figs beam applied analog
advanced bloom first western eastern northern southern national digital universal international royal
super micro intuitive edge fortune sun star eagle coherent lumen compass rocket public service enterprise
waste republic summit pacific atlantic liberty freedom fidelity prudential principal realty energy power
solar wave vital core prime bright clear smart open next new best big true blue green red black white
silver gold diamond crown keystone anchor harbor bridge tower peak mosaic alliance union capital trade
desk data cloud signal vector matrix quantum fusion nexus atlas titan apex zenith vertex pulse spark flex
bumble chewy toast peloton nikola robinhood sea grab wish yum strategy intel
coach ring box dollar riot genius progress premier pinnacle structure viking align pool crispr
twist insight vision focus element harmony avidity exact natural guardant arrowhead ocean ideal
river lake forest mountain rock stone moon sky rain snow wind fire water earth field meadow garden
canyon valley ridge crest harbour haven island coast shore bay cape delta
falcon hawk lion tiger bear bull wolf fox raven phoenix dragon mustang bronco colt stallion cobra viper
panther jaguar puma lynx otter beaver buffalo bison elk moose deer owl swan dove sparrow robin cardinal
kingfisher pelican penguin dolphin whale shark marlin salmon trout bee ant spider scorpion mantis hornet
wasp butterfly mammoth rhino camel zebra gazelle cheetah leopard orca stingray condor kestrel osprey
titan olympus zeus apollo athena atlas hermes mercury jupiter saturn neptune pluto orion nova galaxy
genesis origin catalyst momentum velocity axis alpha beta gamma sigma omega epsilon lambda theta kappa
triumph victory glory noble regal sovereign imperial empire federal standard superior select elite ultra
mega nano pure real perfect simple rapid swift quick fast agile nimble bold brave strong solid steady
stable secure safe trust loyal honest fair just good great grand giant little small mini tiny high low
deep wide long short free live life health care cure heal wellness vita circle sphere cube prism crystal
platinum copper iron steel nickel cobalt lithium uranium carbon graphite oxygen hydrogen helium neon argon
lunar stellar cosmic orbit jet aero air marine coastal arctic polar central legacy heritage horizon
beacon gateway portal cornerstone foundation momentum magnetic electric dynamic kinetic modern classic
adapt achieve advance amplify ascend build create connect deliver drive enable engage evolve expand
explore grow ignite inspire launch lead lift move navigate propel pursue rise shift soar solve thrive
transform unite upgrade venture
bank trust credit capital equity fund hub lab labs works systems network networks solutions services
health medical pharma bio gene genome cell cells protein vaccine clinic hospital
pizza burger coffee tea juice candy chocolate bakery kitchen grill diner tavern
home house living room door window floor wall roof garage yard
car auto motor truck bus train rail ship boat plane drone
book books music movie movies game games play toy toys sport sports
shoe shoes boot boots hat cap shirt dress
paper glass plastic wood metal oil gas coal
north south east west mid upper lower inner outer
one two three four five six seven eight nine ten hundred thousand
news media press times post journal herald tribune daily weekly
old eli dow owl ark hut jet pet net web
duke dominion archer clover hinge devon range plains sunrise sunset dawn
alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho
illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota
mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah
vermont virginia washington wisconsin wyoming carolina dakota jersey york
boston chicago dallas houston austin denver phoenix seattle portland atlanta miami detroit cleveland
pittsburgh philadelphia baltimore charlotte nashville memphis orlando tampa jacksonville vegas
hollywood brooklyn manhattan silicon cupertino
america canada mexico brazil china japan india korea germany france britain london paris berlin
tokyo europe asia africa australia
travelers traveler restaurant restaurants royalty quest trip waters crane icon frontline equitable
encompass tenet carpenter jazz reliance everest monday beyond endeavor planet spire array stem blink
elastic asana hippo paramount fluence tempus diamondback centerpoint atmos vale
cincinnati dover carlisle burlington lincoln hartford edison rogers tyler mueller tyson roper penske motorola
"""
# The last block came from auditing the ~900 largest SEC registrants (2026-10-05): heads that are
# everyday words ("Quest Diagnostics"), places ("Cincinnati Financial"), common surnames
# ("Tyler Technologies", "Tyson Foods") or a louder namesake ("Motorola Solutions" vs the phones).
# Press usage decides the bucket: these are written with their descriptor, so it stays.

COMMON_WORDS: frozenset[str] = frozenset(_RAW.split())

# Brand heads the press *does* use bare ("Axon stock jumps", "Carrier raises outlook", "Nasdaq
# beats") but that collide with ordinary text or a louder namesake in open full-text search (the
# Nasdaq index, Microchip the noun, Charter flights, Serena Williams). They stay the short name for
# finance-context searches; GDELT anchors them like everyday words (legal form, CEO, aliases).
NAMESAKES: frozenset[str] = frozenset("""
charter carrier flutter microchip caesars credo synchrony magna graco axon morningstar williams otis
baxter jacobs rollins woodward ferguson moog brookfield corning fortis nasdaq loews ross
""".split())
