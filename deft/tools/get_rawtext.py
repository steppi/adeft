from indra.literature.elsevier_client import extract_text, logger
from lxml import etree

logger.disabled = True


def extract_text_pubmed(xml):
    tree = etree.fromstring(xml)
    # remove references
    for xref in tree.xpath('//xref'):
        xref.getparent().remove(xref)
    # get paragraphs
    paragraphs = [p.text for p in tree.xpath('//p') if p.text]
    paragraphs = ' '.join(paragraphs)
    paragraphs = ' '.join(paragraphs.split())
    return paragraphs


def extract_text_general(xml):
    try:
        result = extract_text(xml)
    except Exception:
        result = None
    if not result:
        try:
            result = extract_text_pubmed(xml)
        except Exception:
            result = xml
    return result