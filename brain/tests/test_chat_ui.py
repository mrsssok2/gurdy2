import io

from brain.ros.chat_ui import Chat, capitalise_places, redraw_line


def test_capitalise_rooms():
    assert capitalise_places("Found the red box in room a, 1.5 metres ahead.", ["home", "room a", "room b"]) == \
        "Found the red box in Room A, 1.5 metres ahead."
    assert capitalise_places("go room_b then ROOM b", ["room b"]) == "go Room B then Room B"
    assert capitalise_places("a broom a", ["room a"]) == "a broom a"            # whole words only


def test_redraw_keeps_typed_text():
    s = redraw_line("Gurdy: hi", "> ", "go to ro")
    assert s.startswith("\r\x1b[2K") and s.endswith("\n> go to ro")


class _Agent:
    def __init__(self): self.q = [["new object"], []]
    def pump(self): return self.q.pop(0) if self.q else []
    def say(self, t):
        class R: reply = "In room a"
        return R()


def test_async_messages_plain_when_not_a_tty():
    out = io.StringIO()
    c = Chat(_Agent(), lambda: ["room a"], out=out, tty=False, pump_s=0.01)
    c.start()
    import time; time.sleep(0.1); c.stop()
    assert "Gurdy: new object" in out.getvalue()
    assert c.ask("x") == "In Room A"
