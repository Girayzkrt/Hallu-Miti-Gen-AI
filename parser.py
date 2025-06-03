import os
import json
import xml.etree.ElementTree as ET

def extract_text(elem):
    """
    Given an ElementTree element, return all of its text content,
    concatenated and stripped of leading/trailing whitespace.
    If elem is None, return an empty string.
    """
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def extract_title(root):
    """
    Extracts the article title from:
      <front>
        <article-meta>
          <title-group>
            <article-title>...</article-title>
    """
    title_elem = root.find(".//front/article-meta/title-group/article-title")
    return extract_text(title_elem)

def extract_keywords(root):
    """
    Extracts all keywords from:
      <kwd-group>
        <kwd>keyword1</kwd>
        ...
    Returns a list of strings.
    """
    kw_elems = root.findall(".//kwd-group/kwd")
    return [extract_text(kw) for kw in kw_elems]

def extract_abstract(root):
    """
    Extracts the abstract text from:
      <abstract>
        <p>Paragraph 1</p>
        <p>Paragraph 2</p>
        ...
    Joins all <p> tags with newline.
    """
    abstract_elem = root.find(".//abstract")
    if abstract_elem is None:
        return ""
    paras = abstract_elem.findall(".//p")
    texts = [extract_text(p) for p in paras]
    return "\n\n".join(texts).strip()

def extract_section(root, section_name):
    """
    Finds a <sec> whose <title> matches section_name (case-insensitive).
    Then concatenates all <p> tags under that <sec> (including nested).
    Returns the joined text or an empty string if not found.
    """
    target = section_name.strip().lower()

    for sec in root.findall(".//sec"):
        title_elem = sec.find("title")
        if title_elem is not None:
            title_text = extract_text(title_elem).lower()
            if title_text == target or title_text.startswith(target + " " ) or title_text == target + "s":
                paras = sec.findall(".//p")
                texts = [extract_text(p) for p in paras]
                return "\n\n".join(texts).strip()
    return ""

def extract_pmc_id(root):
    """
    Extracts the PMC ID from:
      <article-meta>
        <article-id pub-id-type="pmc">PMCXXXXX</article-id>
    Returns the text or None.
    """
    pmc_elem = root.find(".//front/article-meta/article-id[@pub-id-type='pmc']")
    return pmc_elem.text.strip() if pmc_elem is not None else None

def parse_single_xml(xml_path):
    """
    Parses the XML file, extract required fields, and return a dict:
      {
        "id": <PMC ID or filename>,
        "title": ...,
        "keywords": [...],
        "abstract": ...,
        "introduction": ...,
        "results": ...,
        "conclusion": ...
      }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Failed to parse {xml_path}: {e}")
        return None

    record = {}

    pmc_id = extract_pmc_id(root)
    if pmc_id:
        record["id"] = pmc_id
    else:
        record["id"] = os.path.splitext(os.path.basename(xml_path))[0]

    record["title"] = extract_title(root)
    record["keywords"] = extract_keywords(root)
    record["abstract"] = extract_abstract(root)
    record["introduction"] = extract_section(root, "Introduction")
    record["results"] = extract_section(root, "Results")
    record["conclusion"] = extract_section(root, "Conclusion")

    return record

def parse_pmc_directory(root_dir, output_jsonl_path):
    """
    Walks through all subdirectories of `root_dir`, finds XML files,
    parses each one, and appends its extracted data as a JSON line
    into `output_jsonl_path`.
    """
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for subdir, dirs, files in os.walk(root_dir):
            for filename in files:
                if not filename.lower().endswith(".xml"):
                    continue
                xml_path = os.path.join(subdir, filename)
                record = parse_single_xml(xml_path)
                if record is not None:
                    json_line = json.dumps(record, ensure_ascii=False)
                    out_f.write(json_line + "\n")

if __name__ == "__main__":
    pmc_root = "pmc_extracted"
    output_path = "parsed_pmc.jsonl"

    parse_pmc_directory(pmc_root, output_path)
    print(f"DONE AYO!")
