import json,sys
sys.path.insert(0,'gurdy_core')
from gurdy_core.vision.semantic_map import SemanticMap
L=[json.loads(l) for l in open('docs/results/data/phase5/p56b/o/sightings.jsonl')]
def run(**kw):
    m=SemanticMap(ttl=10.0,min_hits=kw.pop('min_hits',5),min_span=2.0,min_mean_score=kw.pop('ms',0.5),max_scatter=0.5,**kw)
    for s in L: m.update(s['cls'],s['p'],s['score'],s['t'],range_m=s['range'],color=s['color'],weight_scale=0.25 if s['truncated'] else 1.0)
    return sorted((t.cls,t.color,t.hits,round(float(t.pos[0]),1),round(float(t.pos[1]),1)) for t in m.confirmed())
for name,kw in [('old',{}),('dominance 4',dict(dominance_ratio=4.0)),('hits 8 + score 0.6',dict(min_hits=8,ms=0.6)),('both',dict(min_hits=8,ms=0.6,dominance_ratio=4.0))]:
    print(name); [print('  ',x) for x in run(**kw)]
