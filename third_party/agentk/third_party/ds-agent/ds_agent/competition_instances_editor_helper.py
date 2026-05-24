"""
Script to enable easier addition of fields to the competition instances.
- Add new fields in `new_field_map: Dict[competition name: Dict[field_name, field_value]]
- Run the script
- Copy the output
- Paste in competition_instances.py, replacing the content of ALL_COMPETITIONS_LIST


new_field_map = {
    "cassava-leaf-disease-classification": {"new_field1": 1, "new_field2": "blue"},
    "playground-series-s3e11": {"new_field1": 2, "new_field2": "red"},
    "commonlitreadabilityprize": {"new_field1": 3, "new_field2": "green"},
}
"""

from pathlib import Path

from ds_agent.competition_ids import CompetitionID

new_field_map: dict[str, dict[str, ...]] = {}

if __name__ == '__main__':

    script = Path(__file__).parent / "competition_instances.py"

    with script.open() as f:
        content = "".join(f.readlines())
    pattern_to_catch = "ALL_COMPETITIONS_LIST = ["
    content = content[content.find(pattern_to_catch) + len(pattern_to_catch):]
    content = content[:content.find("\n]")].split("Competition(")
    assert content[0].strip() == "", content[0]
    content = content[1:]
    competition_id_list = [getattr(CompetitionID, content[i].split("CompetitionID.")[1].split(",")[0]) for i in
                           range(len(content))]


    def extract_args(body: str) -> str:
        body = body[:body.rfind(")")].strip()
        if body[-1] == ",":
            body = body[:-1]  # remove final comma if there is one --> we only get the body
        return body


    # dictionary of kwrags per competition
    comp_arg_dict = {
        getattr(CompetitionID, content[i].split("CompetitionID.")[1].split(",")[0]): extract_args(content[i]) for i in
        range(len(content))
    }

    for competition_name, new_fields in new_field_map.items():
        comp_id = CompetitionID.get_enum_element(value=competition_name)
        comp_arg_dict[comp_id] += (",\n        " +
                                   ",\n        ".join(map(lambda k: f"{k}={new_fields[k]}", new_fields.keys())))
    s = ""
    for competition_id in competition_id_list:
        s += "    Competition(\n        " + comp_arg_dict[competition_id] + ",\n    ),"
    print(s)
    with open("./tmp_comps.txt", "w") as f:
        f.write(s)
