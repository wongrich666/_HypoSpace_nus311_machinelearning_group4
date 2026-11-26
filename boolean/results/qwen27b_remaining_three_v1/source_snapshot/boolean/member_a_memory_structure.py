from member_a_structure import ROOTS, features, StructureHistory


class MemoryStructureHistory(StructureHistory):

    def choose(self, k=None):
        assert k is None
        history = {r: dict(self.counts[None, r]) for r in ROOTS}
        scores = {
            r: self.counts[None, r]["generated"]
            + self.counts[None, r]["guided"]
            - self.counts[None, r]["compliant_new"]
            for r in ROOTS
        }
        root = min(ROOTS, key=scores.get)
        return {
            "suggested_root": root,
            "scores": scores,
            "history": history,
        }

    def observe(self, k, suggested, row):
        assert k is None
        f = features(row)
        actual = f["root_family"]
        new = bool(row["feedback"]["is_new_valid"])

        if actual:
            cell = self.counts[None, actual]
            cell["generated"] += 1
            cell["new_valid"] += int(new)

        complied = actual == suggested if actual and suggested else None

        if suggested:
            cell = self.counts[None, suggested]
            cell["guided"] += 1
            cell["root_compliant"] += int(complied is True)
            cell["compliant_new"] += int(complied is True and new)

        return {**f, "root_compliance": complied}
