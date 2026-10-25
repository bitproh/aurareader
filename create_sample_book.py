"""
create_sample_book.py - Generates a sample PDF and EPUB book for immediate testing.
"""

import os
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from ebooklib import epub

SAMPLE_CHAPTERS = [
    (
        "Chapter 1: The Cosmic Frontier",
        """The universe is vast beyond ordinary human comprehension. When we gaze into the nocturnal sky, we are looking directly into the deep cosmic past. Light from distant celestial bodies takes millions of years to reach our planet.

Every atom in our bodies was once forged inside the molten core of ancient dying stars. Carbon, oxygen, iron, and calcium were scattered into interstellar space by colossal supernova explosions. We are, in the most literal scientific sense, made of stardust.

Exploring the cosmos challenges our understanding of physics and reality. Robotic rovers now traverse the arid surface of Mars, analyzing ancient lake beds for traces of past microbial life. Space telescopes capture the first moments after the Big Bang with astonishing clarity. As we venture further into the solar system, we discover that every boundary crossed reveals even deeper mysteries waiting to be unraveled."""
    ),
    (
        "Chapter 2: The Quantum Symphony",
        """Beneath the familiar world of everyday experience lies the bizarre realm of quantum mechanics. At subatomic scales, particles behave both as localized packets and spread-out probability waves.

Electrons do not follow neat planetary orbits around atomic nuclei. Instead, they exist in clouds of probability, occupying multiple potential states simultaneously until measured or perturbed. This fundamental indeterminacy is not a failure of our measuring instruments, but an intrinsic principle of nature.

Quantum entanglement links particles across astronomical distances. When two entangled particles are generated, measuring the state of one instantaneously determines the state of the other, regardless of whether they are centimeters or light-years apart. Albert Einstein famously described this phenomenon as spooky action at a distance. Today, quantum principles power modern semiconductors, laser optics, and quantum computing architectures."""
    ),
    (
        "Chapter 3: The Architecture of Mind",
        """Human consciousness remains one of the deepest frontiers in biological science. Inside the human skull rests an intricate neural network composed of roughly eighty-six billion neurons.

Each individual neuron forms thousands of synaptic connections with adjacent cells. Together, they create an electrochemical tapestry capable of abstract thought, emotional resonance, and imaginative foresight. Memories are not static files stored in distinct brain regions, but dynamic patterns of synaptic strength continuously updated.

Cognitive science has shown that perception is an active generative process. The brain does not simply record incoming sensory data; it actively constructs an internal model of external reality. By understanding the neural mechanisms of learning, attention, and memory, humanity takes another step toward unraveling the mystery of self-awareness."""
    )
]


def create_sample_pdf(filepath: str = "sample_book.pdf"):
    doc = fitz.open()
    font_size = 13
    
    for i, (title, content) in enumerate(SAMPLE_CHAPTERS):
        page = doc.new_page(width=595, height=842)  # A4
        # Add Chapter Title
        rect_title = fitz.Rect(50, 50, 545, 90)
        page.insert_textbox(rect_title, title, fontsize=18, fontname="helv", color=(0.1, 0.2, 0.4))
        
        # Add Chapter Body
        rect_body = fitz.Rect(50, 100, 545, 780)
        page.insert_textbox(rect_body, content, fontsize=font_size, fontname="helv", color=(0.1, 0.1, 0.1))

    # Add Table of Contents
    toc = [[1, title, i + 1] for i, (title, _) in enumerate(SAMPLE_CHAPTERS)]
    doc.set_toc(toc)
    
    doc.save(filepath)
    doc.close()
    print(f"Created sample PDF: {filepath}")


def create_sample_epub(filepath: str = "sample_book.epub"):
    book = epub.EpubBook()
    book.set_identifier("sample-book-aurareader-001")
    book.set_title("The Frontier of Science: Cosmos and Mind")
    book.set_language("en")
    book.add_author("AuraReader")

    epub_chapters = []
    spine = ["nav"]

    for i, (title, content) in enumerate(SAMPLE_CHAPTERS):
        ch = epub.EpubHtml(title=title, file_name=f"chap_{i+1}.xhtml", lang="en")
        
        # Format HTML paragraphs
        paras_html = "".join(f"<p>{p.strip()}</p>" for p in content.split("\n\n") if p.strip())
        ch.content = f"""
        <html>
            <head><title>{title}</title></head>
            <body>
                <h1>{title}</h1>
                {paras_html}
            </body>
        </html>
        """
        book.add_item(ch)
        epub_chapters.append(ch)
        spine.append(ch)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine

    epub.write_epub(filepath, book)
    print(f"Created sample EPUB: {filepath}")


if __name__ == "__main__":
    create_sample_pdf()
    create_sample_epub()
