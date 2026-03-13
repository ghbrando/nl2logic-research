from src.preprocessing import EntityLinker


class TestEntityLinker:
    def test_known_acronyms_are_replaced(self):
        linker = EntityLinker()

        linked = linker.link("The OPORD directs the BN into the AO.")

        assert linked == "The Order directs the MilitaryUnit into the Area."

    def test_unknown_tokens_pass_through_unchanged(self):
        linker = EntityLinker()
        nl = "The foobar token remains foobar."

        assert linker.link(nl) == nl

    def test_sumo_terms_already_in_vocabulary_pass_through_unchanged(self):
        linker = EntityLinker()
        nl = "MilitaryOrganization supports MilitaryUnit."

        assert linker.link(nl) == nl

    def test_case_insensitive_matching(self):
        linker = EntityLinker()

        linked = linker.link("The opord supports the bn with c2 in the ao.")

        assert linked == "The Order supports the MilitaryUnit with Communication in the Area."
