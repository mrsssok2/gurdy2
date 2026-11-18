"""SFT data v2 families (docs/results/phase5_eval_ood.md follow-up). Wraps the v1 families with surface variation and adds:

  unknown_object / unknown_capability   decline in plain words, no tool call, never a substitute object
  state_neg                             a place / spot is MENTIONED but nothing may be saved or blocked (the gold has no save_place / no_go_zone)
  resume                                "continue", "go on then" and 20 more ways to resume a paused mission
  follow_other / follow_it              context over the FOCUS slot ("and the other one", "check it out")
  ref_unclear                           "the green thing": no class named -> ask, do not guess
  unknown_place2                        rooms the router does not know -> plan go_to, the safety layer asks

`install()` extends the v1 phrase banks (more casual / terse / indirect templates) and swaps the family table used by
generate.py; `uninstall()` restores v1. Nothing in here is copied from `testset_ood_v1.yaml`: nouns and verbs come from
the lexicons below and generation drops any utterance that is a near paraphrase of a held-out test sentence (generate_v2.py).
"""
from __future__ import annotations

import random

from . import families as F
from .families import C, Ctx, Dialogue, Turn, plan, reply
from .lexicon import OBJECT_WORDS, PLACES, SAVED_PLACE_NAMES, dist_phrase, pick
from .variation import vary

# ---------------------------------------------------------------- lexicons
# things the robot has no concept of. The pre-LLM router (guard.PRE_CHECK_NOUNS) knows about half of them, the model must learn the rest.
UNKNOWN_NOUNS = [
    "table", "desk", "cylinder", "sphere", "ball", "cube", "wall", "window", "couch", "sofa", "bed", "shelf", "bookshelf", "cabinet", "lamp",
    "television", "computer", "laptop", "monitor", "keyboard", "phone", "cup", "mug", "bottle", "plate", "bowl", "book", "bag", "backpack",
    "plant", "tree", "flower", "carpet", "rug", "staircase", "ladder", "fridge", "stove", "sink", "car", "bicycle", "dog", "cat", "stool",
    "bench", "pillar", "rock", "wheel", "screwdriver", "hammer", "bucket", "broom", "curtain", "mirror", "clock", "painting", "pallet", "trolley",
    # not in the router lexicon: the model has to decline these on its own
    "piano", "aquarium", "whiteboard", "projector", "fan", "oven", "microwave", "wardrobe", "forklift", "statue", "fountain", "tent",
    "trash can", "vending machine", "printer", "speaker", "guitar", "umbrella", "suitcase", "skateboard", "helmet", "hat", "shoe", "jacket",
    "toaster", "kettle", "radiator", "fire door", "water cooler", "workbench vise", "generator", "ladder rack", "bin", "sandbag", "tire stack"]
UNKNOWN_PLACES2 = ["reception", "studio", "pantry", "dining room", "storage room", "server room", "gym", "stairwell", "parking lot", "courtyard",
                   "laundry", "nursery", "greenhouse", "loading dock", "mezzanine", "elevator", "restroom", "break room", "conference room"]
ADJ = ["big", "small", "tall", "old", "shiny", "broken", "round", "wooden", "metal", "heavy"]
COLOURS = ["red", "blue", "green", "yellow", "orange", "white"]


def an(n: str) -> str:
    return ("an " if n[0] in "aeiou" else "a ") + n


# ---------------------------------------------------------------- phrase banks added in v2
BANKS_V2 = {
    "GO": ["{t} please", "off to {t}", "i need you at {t}", "get yourself over to {t}", "lets head to {t}", "can we go to {t}", "time to go to {t}",
           "take yourself to {t}", "go {t}", "i would like you to go to {t}", "hop over to {t}", "trot over to {t}", "make a move to {t}", "beeline to {t}"],
    "WALK": ["{d} {m} pls", "step {d} for {m}", "shuffle {d} {m}", "creep {d} {m}", "{m} {d} please", "inch {d} {m}", "{d} by {m}", "can you move {d} {m}",
             "{d}, {m}", "nudge yourself {d} {m}", "i need you {m} {d}"],
    "TURN": ["{d} {a}", "turn {d} by {a}", "spin round {d} {a}", "swivel {d} {a}", "rotate yourself {d} {a}", "twist {d} {a}", "{a} to the {d}",
             "pivot to the {d} {a}", "give me a {a} turn {d}"],
    "FIND": ["find {t} for me", "i lost {t}, find it", "go look for {t}", "hunt down {t}", "can u find {t}", "need {t} found", "look around for {t}",
             "gotta find {t}", "help me find {t}", "i am after {t}", "see where {t} is hiding", "dig up {t}"],
    "Q_WHERE": ["{t} where", "wheres {t}", "where did {t} go", "any idea where {t} is", "i cant find {t}, where is it", "spotted {t} anywhere",
                "know where {t} is", "got eyes on {t}", "where abouts is {t}", "what happened to {t}, where is it"],
    "Q_NEAR": ["whats around", "anything close", "see anything close by", "anything around you", "what is there around", "anything near", "any objects around",
               "whats near", "look around, whats close", "got anything next to you", "what are you standing next to"],
    "Q_BATT": ["battery?", "battery", "how much juice left", "got enough battery", "hows the battery", "u low on power", "power level", "how is your charge",
               "are you running out of power"],
    "Q_DOING": ["what u doing", "whatcha doing", "busy?", "whats going on", "are you doing anything", "doing anything right now", "what are you at",
                "what are you working on"],
    "Q_WHERE_AM": ["where am i at", "whereabouts are you", "whats your location", "where r u", "which place is this", "what spot are you in"],
    "HOME": ["head back", "back to base", "go back to where you started", "return to where you began", "bring yourself home", "home please", "back home",
             "take yourself home", "retreat to the start", "ok go home now", "call it a day and go home"],
    "SAVE": ["remember this place, call it {n}", "rmbr this spot as {n}", "save this as a place named {n}", "bookmark this spot as {n}",
             "make a note of this place as {n}", "save this spot and call it {n}", "store this place under {n}", "mark here as {n}", "name this place {n}",
             "this place should be saved as {n}", "can you remember this location as {n}"],
    "ZONE_ADD": ["stay away from {p}", "stay out of {p}", "dont ever go into {p}", "{p} is off limits", "{p} is a no go area", "never ever enter {p}",
                 "keep clear of {p}", "steer clear of {p}", "{p} is dangerous, stay out", "no entry to {p}", "stay off {p}", "block {p} for yourself",
                 "do not walk into {p}"],
    "ZONE_REMOVE": ["{p} is safe again", "you can go into {p} again", "unblock {p} please", "lift the restriction on {p}", "forget the no-go zone at {p}",
                    "{p} is not forbidden anymore", "reopen {p}", "remove the block on {p}"],
    "INSPECT": ["is {t} still there", "go see if {t} is still there", "swing by {t} and check", "pop over and look at {t}", "give {t} a once over"],
    "EXPLORE": ["look around the place", "go see whats around", "wander around and map it", "check out the surroundings", "take a tour and map things",
                "go map stuff"],
}
_SAVED: dict = {}


def install() -> None:
    for name, extra in BANKS_V2.items():
        cur = getattr(F, name)
        _SAVED[name] = list(cur)
        cur.extend(extra)
    import brain.data_gen.generate as G
    _SAVED["_FAMILIES"] = G.FAMILIES
    G.FAMILIES = FAMILIES_V2


def uninstall() -> None:
    for name, old in list(_SAVED.items()):
        if name == "_FAMILIES":
            import brain.data_gen.generate as G
            G.FAMILIES = old
        else:
            getattr(F, name)[:] = old
    _SAVED.clear()


# ---------------------------------------------------------------- wrappers
def varied(fn, p: float = 0.8):
    """A v1 family whose user sentences get a casual / typo / slang / indirect surface form (gold unchanged)."""
    def g(c: Ctx):
        d = fn(c)
        if d is None:
            return None
        for t in d.turns:
            if not t.bypass_agent:
                t.say = vary(c.rng, t.say, p)
        return d
    g.__name__ = "v_" + fn.__name__
    return g


def _bypass(say: str, text: str) -> Turn:
    return Turn(say, [reply(text)], ("refuse", "none"), bypass_agent=True)


# ---------------------------------------------------------------- new families
DECLINE_OBJ = ["I can't do that. I don't know what {a} is, and I have not seen one.",
               "Sorry, I don't know what {a} is, so I can't help with that.",
               "I have not seen {a}, and I don't know that kind of object.",
               "I can't find {a}. I don't know that kind of object.",
               "I don't know {a}. I only know the objects my camera has been trained on, so I can't do that.",
               "No, I can't. {A} is not something I know about, and I have not seen one."]
OBJ_INTENTS = ["where is the {n}", "find the {n}", "have you seen the {n}", "is there {a} around", "inspect the {n}", "what about the {n}",
               "how far is the {n}", "look for the {n}", "where did you last see the {n}", "do you remember the {n}", "locate the {n} on the map",
               "check the {n}", "is the {n} still there", "the {n}, where is it", "can you see the {n}", "tell me about the {n}", "search for {a}",
               "scan for the {n}", "go and look at the {n}", "have a look at the {n}", "any {n} nearby", "wheres the {n}", "whats that {n} doing there",
               "avoid the {n}", "stay away from the {n}"]


def f_unknown_object(c: Ctx):
    rng = c.rng
    n = rng.choice(UNKNOWN_NOUNS)
    if rng.random() < 0.2:
        n = f"{rng.choice(COLOURS + ADJ)} {n}"
    tmpl = rng.choice(OBJ_INTENTS)
    say = tmpl.format(n=n, a=an(n))
    text = rng.choice(DECLINE_OBJ).format(a=an(n), A=an(n).capitalize())
    if rng.random() < 0.25:               # second turn after a normal answer
        objs = c.unique_seen()
        if objs:
            o = rng.choice(objs)
            t, ref = c.obj_phrase(o, force_color=True)
            t1 = Turn(F._say(c, F.Q_WHERE, True, t=t), [plan(C("where_is", object=ref))], ("answer",))
            t2 = _bypass(vary(rng, rng.choice(["and what about the {n}", "how about the {n}", "and the {n}", "what about the {n} then"]).format(n=n)), text)
            return Dialogue("unknown_object", [t1, t2])
    return Dialogue("unknown_object", [_bypass(vary(rng, say), text)])


CAPABILITY = {
    "carry": (["carry the {o} to room b", "fetch the {o}", "pick up the {o}", "grab the {o}", "lift the {o}", "deliver the {o} to me",
               "put the {o} in room a", "take the {o} with you", "hand me the {o}", "move the {o} out of the way", "tidy up the {o}"],
              ["I can't do that. I cannot pick up or carry things.", "Sorry, I can't carry or move objects. I can only go near them.",
               "I can't pick things up. I can walk to an object, but I cannot bring it to you."]),
    "world": (["open the door", "close the window", "turn on the lights", "switch off the lamp", "clean the floor", "make me a coffee", "water the plants",
               "wipe the table", "press the button", "unlock the door", "plug yourself in", "do the dishes"],
              ["I can't do that. I have no arms or hands to act on things.", "Sorry, I can't do that. I can only move around and look at things.",
               "I can't do that. I am a walking robot without a way to operate things."]),
    "media": (["play some music", "sing me a song", "take a photo", "record a video", "send a message to my friend", "call my brother", "set an alarm",
               "what is the weather", "what time is it", "read me the news", "order a pizza", "dance for me", "tell me a story"],
              ["I can't do that. I can move around, find objects, explore, patrol and answer questions about what I have seen.",
               "Sorry, that is outside what I can do. I can move, look for objects and report what I know.",
               "I can't help with that. I only handle moving around and what I have seen."]),
    "body": (["jump", "fly up to the ceiling", "climb the stairs", "swim across", "do a backflip", "stand on two legs", "go faster than light", "lift a leg",
              "wave at me", "shake hands"],
             ["I can't do that. My body cannot do that kind of move.", "Sorry, I can't. I can only walk and turn.",
              "I can't do that. I can only walk and turn on the spot."]),
    "settings": (["walk at double speed", "slow down a bit", "set your speed to max", "change your gait", "make your legs stronger", "raise your height",
                  "switch to night mode", "reboot yourself", "update your software", "use a different map"],
                 ["I can't do that. I don't have a setting for it.", "Sorry, I can't change that about myself.",
                  "I can't do that. That is not something I can change."]),
}


def f_unknown_capability(c: Ctx):
    rng = c.rng
    kind = rng.choice(list(CAPABILITY))
    tmpls, replies = CAPABILITY[kind]
    o = rng.choice(["red box", "blue box", "barrel", "toolbox", "cone", "extinguisher", "green barrel", "yellow box"])
    say = rng.choice(tmpls).format(o=o)
    return Dialogue("unknown_capability", [_bypass(vary(rng, say), rng.choice(replies))])


COMMENT_PLACE = ["{p} looks nice today", "{p} is a mess", "i left my keys in {p}", "{p} is cold", "i like {p}", "everybody is in {p}", "{p} is pretty big",
                 "its quiet in {p}", "someone mopped {p}", "the light in {p} is on", "{p} was cleaned yesterday"]
COMMENT_HERE = ["this spot is nice", "what a nice place", "this place is a bit dusty", "i like it here", "it is bright here", "good spot this one", "cozy in here",
                "we are at the workbench now", "this is where i usually sit", "i am standing right next to you"]


def f_state_neg(c: Ctx):
    """Mentions a place / spot as small talk and then asks for something else: the gold must not save a place or block an area."""
    rng = c.rng
    p = rng.choice(["room A", "room B"])
    k = rng.choice(["place_go", "here_go", "here_q", "order", "order"])
    sep = rng.choice([", ", ". ", ", so ", ". now ", " but "])
    if k == "place_go":
        q = rng.choice(["room A", "room B", "the doorway"])
        if rng.random() < 0.5:
            q = ("room A" if p == "room B" else "room B")
        tgt = ("doorway" if q == "the doorway" else q)
        say = rng.choice(COMMENT_PLACE).format(p=p) + sep + pick(rng, F.GO, c.split).format(t=q)
        return Dialogue("state_neg", [Turn(vary(rng, say, 0.5), [plan(C("go_to", place=tgt))], ("execute", "refuse", "confirm"))])
    if k == "here_go":
        say = rng.choice(COMMENT_HERE) + sep + pick(rng, F.GO, c.split).format(t=p)
        return Dialogue("state_neg", [Turn(vary(rng, say, 0.5), [plan(C("go_to", place=p))], ("execute", "refuse", "confirm"))])
    if k == "here_q":
        say = rng.choice(COMMENT_HERE) + sep + rng.choice(["where are we", "which room is this", "how far is home", "where am i"])
        return Dialogue("state_neg", [Turn(vary(rng, say, 0.5), [plan(C("where_am_i"))], ("answer",))])
    # ordering: inspect something, then go somewhere, no save
    objs = c.unique_seen()
    if not objs:
        return None
    o = rng.choice(objs)
    t, ref = c.obj_phrase(o, force_color=True)
    home = rng.random() < 0.5
    dest = "home" if home else p
    last = C("return_home") if home else C("go_to", place=p)
    say = rng.choice(["first inspect {t} then {d}", "check {t} out and then {d}", "{d} once you have looked at {t}", "before {d}, inspect {t}",
                      "when {t} has been checked, {d}"]).format(t=t, d=("go back to the start" if home else f"go to {p}"))
    return Dialogue("state_neg", [Turn(vary(rng, say, 0.5), [plan(C("inspect", object=ref), last)], ("execute", "refuse", "confirm", "clarify"))])


RESUME = ["continue", "go on then", "keep going", "ok continue", "resume", "unpause", "continue please",
          "go ahead and continue", "pick up where you left off", "proceed", "back to it", "continue with what you were doing", "you can go again", "ok go on", "ready when you are, continue", "alright resume the mission", "get going again", "off you go again",
          "ok that is enough of a break", "lets continue", "keep at it", "resume the task", "start moving again"]
PAUSE = ["pause", "hold on a sec", "pause for a moment", "wait a moment", "pause the mission", "hang on", "pause it please",
         "freeze the task for now"]
CANCEL = ["cancel the mission", "abort", "forget that", "never mind the task", "cancel that", "drop it", "scrap that mission", "i changed my mind, cancel"]


def f_resume(c: Ctx):
    rng = c.rng
    c.need.update({"explored": True, "pose": {"place": "home", "yaw_deg": 0}, "battery": 85.0, "fail_legs": []})   # a long, allowed mission to pause
    p = "room B"
    t1 = Turn(F._say(c, F.GO, t=p), [plan(C("go_to", place=p))], ("execute", "refuse", "confirm"), wait_s=6)
    act = rng.choice(["resume", "resume", "resume", "pause", "cancel"])
    turns = [t1]
    if act == "resume":
        turns.append(Turn(vary(rng, rng.choice(PAUSE)), [plan(C("mission_control", action="pause"))], ("execute",), wait_s=2))
    bank = {"resume": RESUME, "pause": PAUSE, "cancel": CANCEL}[act]
    turns.append(Turn(vary(rng, rng.choice(bank), 0.6), [plan(C("mission_control", action=act))], ("execute", "answer", "none")))
    return Dialogue("resume", turns)


def f_follow_other(c: Ctx):
    """'and the other box': the model sees only the focus object, so it asks for the whole class; the tool layer lists them / asks which."""
    rng = c.rng
    cnt: dict[str, list] = {}
    for t in c.seen():
        cnt.setdefault(t["class"], []).append(t)
    multi = [k for k, v in cnt.items() if len(v) >= 2]
    if not multi:
        return None
    cls = rng.choice(multi)
    o = rng.choice([t for t in cnt[cls] if sum(1 for u in cnt[cls] if u["color"] == t["color"]) == 1] or cnt[cls])
    t, ref = c.obj_phrase(o, force_color=True)
    word = rng.choice(OBJECT_WORDS[cls])
    t1 = Turn(F._say(c, F.Q_WHERE, True, t=t), [plan(C("where_is", object=ref))], ("answer",))
    if rng.random() < 0.55:
        t2 = Turn(vary(rng, rng.choice(["and the other {w}", "what about the other {w}", "how about the second {w}", "and where is the other one {w}",
                                         "the other {w} then"]).format(w=word), 0.6),
                  [plan(C("where_is", object={"class": cls}))], ("answer",))
    else:
        t2 = Turn(vary(rng, rng.choice(["go to the other {w}", "now the other {w}", "take me to the other {w} instead", "i meant the other {w}"]).format(w=word), 0.6),
                  [plan(C("go_to", object={"class": cls}))], ("clarify", "execute", "confirm", "refuse"))
    return Dialogue("follow_other", [t1, t2])


def f_follow_it(c: Ctx):
    rng = c.rng
    objs = c.unique_seen()
    if not objs:
        return None
    o = rng.choice(objs)
    t, ref = c.obj_phrase(o, force_color=True)
    t1 = Turn(F._say(c, F.Q_WHERE, True, t=t), [plan(C("where_is", object=ref))], ("answer",))
    k = rng.choice(["inspect", "path", "find", "go"])
    if k == "inspect":
        t2 = Turn(vary(rng, rng.choice(["check it out", "have a look at that", "is it still there", "make sure it is still there", "look at it up close",
                                         "go see it", "verify that one"]), 0.6), [plan(C("inspect", object=ref))], ("execute", "refuse", "confirm"))
    elif k == "path":
        t2 = Turn(vary(rng, rng.choice(["is it far", "how long to get there", "can you get to it", "would it take long to reach", "is the path clear"]), 0.6),
                  [plan(C("path_check", object=ref))], ("answer",))
    elif k == "find":
        t2 = Turn(vary(rng, rng.choice(["go look for it again", "find that again", "search for it again", "hunt for that one"]), 0.6),
                  [plan(C("find", object=ref))], ("execute", "refuse", "confirm"))
    else:
        t2 = Turn(vary(rng, rng.choice(["go over to it", "take me there", "walk to that one", "head to it"]), 0.6), [plan(C("go_to", object=ref))],
                  ("execute", "refuse", "confirm"))
    return Dialogue("follow_it", [t1, t2])


REF_VAGUE = ["the {col} thing, take me there", "find something {col}", "head for the {col} one", "i want to see the {col} object", "go to that {col} item",
             "show me anything {col}", "i am looking for something {col}", "the {adj} {col} thing, go there", "go near the {adj} one",
             "where is the {adj} thing", "inspect that {adj} item", "go to the {adj} object", "bring up the {col} stuff", "which one is {col}, go there"]


def f_ref_unclear(c: Ctx):
    rng = c.rng
    say = rng.choice(REF_VAGUE).format(col=rng.choice(COLOURS), adj=rng.choice(ADJ))
    q = rng.choice(["Which kind of object do you mean?", "What kind of object is it?", "Which object do you mean? Tell me what kind it is.",
                    "Could you tell me what kind of object you mean?"])
    return Dialogue("ref_unclear", [Turn(vary(rng, say, 0.7), [plan(C("ask_user", question=q))], ("clarify",))])


def f_unknown_place2(c: Ctx):
    rng = c.rng
    p = rng.choice(UNKNOWN_PLACES2)
    say = pick(rng, F.GO, c.split).format(t=("the " + p) if rng.random() < 0.6 else p)
    return Dialogue("unknown_place2", [Turn(vary(rng, say, 0.7), [plan(C("go_to", place=p))], ("clarify",))])


# ---------------------------------------------------------------- table
def _wrapped_v1() -> list:
    out = []
    for w, g, fn in F.FAMILIES:
        if fn is F.f_unsafe_text:                       # regex fast path: untouched
            out.append((w, g, fn))
        elif fn in (F.f_zone, F.f_save_place):
            out.append((w * 2.5, g, varied(fn)))        # explicit state changes, many more phrasings
        else:
            out.append((w * 0.62, g, varied(fn, 0.85)))
    return out


FAMILIES_V2: list = _wrapped_v1() + [
    (14, "unknown", f_unknown_object), (9, "unknown", f_unknown_capability), (5, "unknown", f_unknown_place2),
    (13, "state_neg", f_state_neg), (8, "multiturn", f_resume), (8, "multiturn", f_follow_other), (6, "multiturn", f_follow_it),
    (7, "behaviour", f_ref_unclear),
]
