"""Run: python3 -m unittest ops.tests.test_incident_store  (also collected by pytest). No ROS, no Docker."""
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from ops.incidents import Cause, Event, IncidentStore, Run, fingerprint


def ev(t, type_="leg_failure", sev="critical", text="leg 3 tracking error 0.21 rad", run="r1", **kw):
    return Event(run, t, type_, sev, "alert_monitor", text, **kw)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.s = IncidentStore(":memory:", wall=lambda: 1000.0)
        self.s.add_run(Run("r1", "abc", "terrain_dr2", "disabled_leg", {"kd": 0.05}))
        self.s.add_run(Run("r2"))

    def test_run_roundtrip_and_duplicate(self):
        r = self.s.get_run("r1")
        self.assertEqual((r.controller, r.fault_label, r.config, r.created), ("terrain_dr2", "disabled_leg", {"kd": 0.05}, 1000.0))
        self.assertIsNone(self.s.get_run("nope"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.s.add_run(Run("r1"))
        self.assertEqual([r.run_id for r in self.s.runs()], ["r1", "r2"])

    def test_event_roundtrip_filters_and_fk(self):
        a = self.s.add_event(ev(5.0, data={"leg": 3}))
        self.s.add_event(ev(9.0, "fall", "critical", "robot fell"))
        e = self.s.get_event(a)
        self.assertEqual((e.id, e.data, e.severity), (a, {"leg": 3}, "critical"))
        self.assertEqual(len(self.s.events("r1")), 2)
        self.assertEqual([x.type for x in self.s.events("r1", since=6.0)], ["fall"])
        self.assertEqual([x.type for x in self.s.events("r1", type="leg_failure")], ["leg_failure"])
        with self.assertRaises(sqlite3.IntegrityError):
            self.s.add_event(ev(1.0, run="ghost"))
        with self.assertRaises(ValueError):
            Event("r1", 1.0, "x", "fatal", "s")

    def test_fingerprint_ignores_numbers(self):
        self.assertEqual(fingerprint("t", "s", "leg 3 error 0.21 rad"), fingerprint("t", "s", "Leg 5  error 0.34 rad"))
        self.assertNotEqual(fingerprint("t", "s", "leg error"), fingerprint("u", "s", "leg error"))
        self.s.add_event(ev(1.0))
        self.s.add_event(ev(2.0, text="leg 5 tracking error 0.40 rad"))
        self.assertEqual(list(self.s.fingerprint_counts("r1").values()), [2])

    def test_incident_severity_end_and_close(self):
        i = self.s.open_incident("r1", 5.0, "warning", "slipping")
        a = self.s.add_event(ev(5.0, "stuck", "warning", "stuck"))
        b = self.s.add_event(ev(12.0))
        c = self.s.add_event(ev(40.0, "estop", "critical", "operator stop"))
        self.s.add_event_to_incident(i, a, "trigger")
        self.s.add_event_to_incident(i, b)
        self.assertEqual(self.s.get_incident(i).severity, "critical")
        self.s.add_event_to_incident(i, c, "context")          # context does not move severity, but still extends the window
        inc = self.s.get_incident(i)
        self.assertEqual((inc.event_ids, inc.end), ([a, b, c], 40.0))
        self.s.close_incident(i, end=30.0)                      # never shrinks below the last member
        self.assertEqual((self.s.get_incident(i).status, self.s.get_incident(i).end), ("closed", 40.0))
        with self.assertRaises(ValueError):
            self.s.add_event_to_incident(i, self.s.add_event(ev(50.0)))

    def test_context_event_does_not_raise_severity(self):
        i = self.s.open_incident("r1", 1.0, "warning")
        self.s.add_event_to_incident(i, self.s.add_event(ev(1.0, "estop", "critical")), "context")
        self.assertEqual(self.s.get_incident(i).severity, "warning")

    def test_membership_errors(self):
        i = self.s.open_incident("r1", 1.0, "info")
        e = self.s.add_event(ev(1.0, run="r2"))
        with self.assertRaises(ValueError):                    # other run
            self.s.add_event_to_incident(i, e)
        e1 = self.s.add_event(ev(2.0))
        self.s.add_event_to_incident(i, e1)
        with self.assertRaises(sqlite3.IntegrityError):        # same event twice
            self.s.add_event_to_incident(i, e1)
        with self.assertRaises(KeyError):
            self.s.add_event_to_incident(i, 999)
        with self.assertRaises(ValueError):
            self.s.add_event_to_incident(i, e1, "boss")
        with self.assertRaises(ValueError):
            self.s.open_incident("r1", 1.0, "bad")

    def test_incident_filters(self):
        a = self.s.open_incident("r1", 1.0, "warning")
        b = self.s.open_incident("r1", 50.0, "critical")
        c = self.s.open_incident("r2", 3.0, "info")
        self.s.close_incident(a, 10.0)
        ids = lambda **kw: [i.id for i in self.s.incidents(**kw)]
        self.assertEqual(ids(), [a, c, b])
        self.assertEqual(ids(run_id="r1"), [a, b])
        self.assertEqual(ids(status="open"), [c, b])
        self.assertEqual(ids(min_severity="warning"), [a, b])
        self.assertEqual(ids(since=20.0), [c, b])               # a ended at 10; open ones always count
        with self.assertRaises(ValueError):
            self.s.incidents(status="weird")

    def test_causes_ranked_with_evidence_and_validation(self):
        i = self.s.open_incident("r1", 1.0, "critical")
        weak = self.s.add_cause(i, "weak motors", 0.3, "rule")
        top = self.s.add_cause(i, "disabled leg", 0.8, "rule")
        self.s.add_evidence(top, "metric", "leg3_track_err@10s", "9x the other legs")
        self.s.add_evidence(top, "event", "1", "leg_failure alert")
        cs = self.s.causes(i)
        self.assertEqual([c.label for c in cs], ["disabled leg", "weak motors"])
        self.assertEqual([e.kind for e in cs[0].evidence], ["metric", "event"])
        self.assertEqual(cs[1].evidence, [])
        for bad in (lambda: self.s.add_cause(i, "x", 1.5, "rule"), lambda: self.s.add_cause(i, "x", 0.5, "llm"),
                    lambda: self.s.add_evidence(top, "gossip", "r", "s")):
            with self.assertRaises(ValueError):
                bad()
        with self.assertRaises(KeyError):
            self.s.add_cause(999, "x", 0.5, "rule")
        with self.assertRaises(KeyError):
            self.s.add_evidence(999, "log", "r", "s")
        self.assertIsInstance(cs[0], Cause)
        self.assertGreater(weak, 0)

    def test_similar_and_detail(self):
        a = self.s.open_incident("r1", 1.0, "critical", "leg 3")
        b = self.s.open_incident("r2", 1.0, "critical")
        c = self.s.open_incident("r2", 9.0, "info")
        self.s.set_similar(a, b, 0.9)
        self.s.set_similar(a, c, 0.4)
        self.s.set_similar(a, c, 0.5)                           # upsert
        self.assertEqual(self.s.similar(a), [(b, 0.9), (c, 0.5)])
        self.assertEqual(self.s.similar(a, k=1), [(b, 0.9)])
        with self.assertRaises(ValueError):
            self.s.set_similar(a, a, 1.0)
        self.s.add_event_to_incident(a, self.s.add_event(ev(2.0)))
        cause = self.s.add_cause(a, "disabled leg", 0.8, "label")
        self.s.add_evidence(cause, "log", "line 4", "leg 3 limp")
        d = self.s.detail(a)
        self.assertEqual((d["n_events"], d["events_by_type"], d["causes"][0]["evidence"][0]["summary"]),
                         (1, {"leg_failure": 1}, "leg 3 limp"))
        self.assertEqual(d["similar"][0]["incident_id"], b)
        self.assertIsNone(self.s.detail(999))
        self.assertIsNone(self.s.get_incident(999))


class PersistAndThreads(unittest.TestCase):
    def test_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "inc.db"
            s = IncidentStore(p)
            s.add_run(Run("r1"))
            i = s.open_incident("r1", 1.0, "warning", "t")
            s.add_cause(i, "chatter", 0.7, "rule")
            s.close()
            s2 = IncidentStore(p)
            self.assertEqual(s2.get_incident(i).title, "t")
            self.assertEqual(s2.causes(i)[0].label, "chatter")
            s2.close()

    def test_used_from_other_threads(self):
        s = IncidentStore(":memory:")
        s.add_run(Run("r1"))
        errors, ids = [], []

        def work(n):
            try:
                for k in range(25):
                    ids.append(s.add_event(Event("r1", float(n * 100 + k), "stuck", "warning", "t", f"e {k}")))
                    s.events("r1")
            except Exception as e:      # noqa: BLE001 - the test reports any failure
                errors.append(e)

        ts = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errors, [])
        self.assertEqual((len(ids), len(set(ids)), len(s.events("r1"))), (150, 150, 150))


if __name__ == "__main__":
    unittest.main()
