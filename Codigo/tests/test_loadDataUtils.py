# def test_get_target_id_maximum_activity():
#     # SAMPLE DATA
#     organism_name = "Organism A"
#     activity_type = "IC50"
#     targets_id = ["target1", "target2", "target3"]
#     molecules_data = {
#         "target_chembl_id": ["target1", "target1", "target2", "target3", "target3"],
#         "molecule_chembl_id": [
#             "molecule1",
#             "molecule2",
#             "molecule3",
#             "molecule4",
#             "molecule5",
#         ],
#     }
#     molecules = pd.DataFrame.from_dict(molecules_data)

#     # MOCKS CLIENTS
#     class MockTargetClient:
#         def filter(self, organism):
#             return self

#         def only(self, fields):
#             return [{"target_chembl_id": target_id} for target_id in targets_id]

#     class MockActivityClient:
#         def filter(self, target_chembl_id, standard_type):
#             return self

#         def only(self, fields):
#             return molecules_data

#     # REPLACE CLIENTS
#     chembl.get_target_id_maximum_activity.target_client = MockTargetClient()
#     chembl.get_target_id_maximum_activity.activity_client = MockActivityClient()

#     # CALL FUNCTION
#     result = chembl.get_target_id_maximum_activity(organism_name, activity_type)

#     # TEST THE RESULT
#     assert result == "target1"
