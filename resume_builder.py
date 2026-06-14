"""
Smart Interview and PDF Generator for ResumeIQ.

Analyzes ML pipeline output to generate targeted interview questions,
then builds a clean, professional PDF resume from the user's answers
and original resume text.
"""

from fpdf import FPDF
import re
import os


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_name(text: str) -> str:
    """Return the first non-empty line of *text* as the candidate's name."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "Your Name"


def _extract_section(text: str, section_name: str) -> str:
    """Try to extract the content between *section_name* and the next section header.

    Section headers are identified as lines that look like all-caps labels,
    title-case labels, or lines ending with a colon.
    """
    # Build a pattern that matches the section header (case-insensitive)
    header_pattern = re.compile(
        rf"(?i)^\s*{re.escape(section_name)}\s*:?\s*$", re.MULTILINE
    )
    match = header_pattern.search(text)
    if not match:
        return ""

    start = match.end()

    # Look for the next section-like header after the matched one
    next_header = re.compile(
        r"^\s*[A-Z][A-Za-z &/]+\s*:?\s*$", re.MULTILINE
    )
    next_match = next_header.search(text, start)
    end = next_match.start() if next_match else len(text)

    return text[start:end].strip()


def _safe_text(text: str) -> str:
    """Replace non-latin characters so FPDF can render the string."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# 1. Interview-question generator
# ---------------------------------------------------------------------------

def generate_interview_questions(
    issues: list[str],
    skills: list[str],
    market_gaps: dict,
    bullet_results: list[dict],
) -> list[str]:
    """Analyze ML output and dynamically generate 5-7 targeted questions.

    Parameters
    ----------
    issues : list[str]
        List of issue strings detected by the resume analyzer
        (e.g. ``["No email found", "No Experience section"]``).
    skills : list[str]
        Skills already present on the resume.
    market_gaps : dict
        Dictionary with at least a ``"missing_skills"`` key whose value is a
        list of in-demand skills the candidate lacks.
    bullet_results : list[dict]
        Per-bullet analysis dicts, each containing at least a ``"label"`` key
        (``"Strong"`` or ``"Weak"``).

    Returns
    -------
    list[str]
        Between 5 and 7 question strings.
    """
    questions: list[str] = []

    # Extract the 'issue' text from each issue dictionary
    issues_lower = [i.get("issue", "").lower() for i in issues]
    issues_text = " ".join(issues_lower)

    contact_missing = any(
        kw in issues_text
        for kw in ("no email", "no phone", "contact", "missing email", "missing phone")
    )
    if contact_missing:
        questions.append(
            "What is your email address and phone number?"
        )

    linkedin_missing = any("linkedin" in issue for issue in issues_lower)
    if linkedin_missing:
        questions.append(
            "What is your LinkedIn profile URL?"
        )

    # --- Experience section ----------------------------------------------------
    if any("no experience" in issue for issue in issues_lower):
        questions.append(
            "Please describe your most recent work experience or internship, "
            "including your role, company name, and key responsibilities."
        )

    # --- Metrics / measurable results ------------------------------------------
    metrics_lacking = any(
        kw in issue for issue in issues_lower
        for kw in ("metric", "quantif", "measurable", "no numbers")
    )
    if metrics_lacking:
        questions.append(
            "Can you describe a project where you achieved a measurable result? "
            "(e.g., Increased sales by 20%, Reduced load time by 40%)"
        )

    # --- Action verbs ----------------------------------------------------------
    weak_verbs = any(
        kw in issue for issue in issues_lower
        for kw in ("action verb", "weak verb", "passive")
    )
    if weak_verbs:
        questions.append(
            "Describe your biggest professional achievement using strong action "
            "words (e.g., Led, Developed, Optimized)."
        )

    # --- Market-gap skills -----------------------------------------------------
    missing_skills = market_gaps.get("missing_skills", [])
    if missing_skills:
        missing_skills_str = ", ".join(missing_skills[:6])
        questions.append(
            f"Your profile is missing in-demand skills like {missing_skills_str}. "
            "Do you have experience with any of these? If yes, briefly describe."
        )

    # --- Weak bullet points ----------------------------------------------------
    has_weak = any(
        b.get("label", "").lower() == "weak" for b in bullet_results
    )
    if has_weak:
        questions.append(
            "Some of your bullet points lack impact. Rewrite your most important "
            "accomplishment in one powerful sentence."
        )

    # --- Pad to at least 5 questions ------------------------------------------
    generic_questions = [
        "In 2-3 sentences, write a professional summary/objective for your resume.",
        "List any certifications, awards, or notable projects not already on your resume.",
        "What are your top three technical or professional strengths?",
        "Describe a challenge you overcame at work and the outcome.",
        "What career goals would you like your resume to reflect?",
    ]

    for gq in generic_questions:
        if len(questions) >= 5:
            break
        if gq not in questions:
            questions.append(gq)

    # Cap at 7
    return questions[:7]


# ---------------------------------------------------------------------------
# 2. Enhanced PDF generator
# ---------------------------------------------------------------------------

def generate_enhanced_pdf(
    original_text: str,
    answers: dict[str, str],
    skills: list[str],
    predicted_role: str,
    output_path: str = "enhanced_resume.pdf",
) -> str:
    """Generate a clean, professional PDF resume.

    Parameters
    ----------
    original_text : str
        The raw text of the candidate's original resume.
    answers : dict[str, str]
        Mapping of question → answer collected during the interview step.
    skills : list[str]
        Aggregated list of skills (extracted + user-provided).
    predicted_role : str
        The role predicted by the ML classifier (used as a subtitle hint).
    output_path : str, optional
        Destination file path for the PDF. Defaults to ``'enhanced_resume.pdf'``.

    Returns
    -------
    str
        The absolute path of the generated PDF file.
    """

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # -- Colour constants ------------------------------------------------------
    DARK_BLUE = (44, 62, 80)
    BLACK = (0, 0, 0)

    # -- Extract core details --------------------------------------------------
    name = _safe_text(_extract_name(original_text))

    # Try to pull email and phone from original text first
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", original_text)
    phone_match = re.search(r"[\+]?[\d\s\-().]{7,15}", original_text)

    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0).strip() if phone_match else ""

    # Override with interview answers if the user provided contact info
    for q, a in answers.items():
        q_lower = q.lower()
        if "email" in q_lower and "phone" in q_lower and a.strip():
            # Try to split email and phone from the answer
            ans_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", a)
            ans_phone = re.search(r"[\+]?[\d\s\-().]{7,15}", a)
            if ans_email:
                email = ans_email.group(0)
            if ans_phone:
                phone = ans_phone.group(0).strip()

    contact_parts = [p for p in (email, phone) if p]
    contact_line = _safe_text(" | ".join(contact_parts)) if contact_parts else ""

    # -- Title (Name) ----------------------------------------------------------
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 10, name.upper(), new_x="LMARGIN", new_y="NEXT", align="C")

    if contact_line:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, contact_line, new_x="LMARGIN", new_y="NEXT", align="C")

    if predicted_role:
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, _safe_text(predicted_role), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_text_color(*BLACK)

    pdf.ln(2)
    # Header separator line
    pdf.set_draw_color(*BLACK)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # -- Helper: add a section -------------------------------------------------
    def _add_section(title: str, body: str) -> None:
        """Render a section header + body text block with proper ATS formatting."""
        if not body.strip():
            return

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 7, _safe_text(title.upper()), new_x="LMARGIN", new_y="NEXT")

        # Thin separator line
        pdf.set_draw_color(*BLACK)
        pdf.set_line_width(0.2)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(*BLACK)
        
        # Parse text into bullet points and paragraphs
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                # Clean bullet
                clean_line = _safe_text(line.lstrip("-•* ").strip())
                pdf.set_x(20) # Indent for bullet
                pdf.multi_cell(0, 5, "\x95  " + clean_line) # \x95 is FPDF bullet
            else:
                pdf.set_x(15)
                pdf.multi_cell(0, 5.5, _safe_text(line))
                
        pdf.ln(4)

    # -- Professional Summary --------------------------------------------------
    summary_answer = ""
    for q, a in answers.items():
        if "summary" in q.lower() or "objective" in q.lower():
            summary_answer = a.strip()
            break

    if not summary_answer:
        # Fall back to the first paragraph of the original text
        paragraphs = [p.strip() for p in original_text.split("\n\n") if p.strip()]
        summary_answer = paragraphs[0] if paragraphs else ""

    _add_section("Professional Summary", summary_answer)

    # -- Skills ----------------------------------------------------------------
    skills_text = ", ".join(skills) if skills else ""
    _add_section("Skills", skills_text)

    # -- Experience ------------------------------------------------------------
    experience_body = _extract_section(original_text, "Experience")
    # Append answer about experience if provided
    for q, a in answers.items():
        if "experience" in q.lower() or "internship" in q.lower():
            if a.strip():
                experience_body = (
                    f"{experience_body}\n\n{a.strip()}" if experience_body else a.strip()
                )
            break
    _add_section("Experience", experience_body)

    # -- Education -------------------------------------------------------------
    education_body = _extract_section(original_text, "Education")
    _add_section("Education", education_body)

    # -- Achievements ----------------------------------------------------------
    achievements_parts: list[str] = []
    for q, a in answers.items():
        q_lower = q.lower()
        if any(kw in q_lower for kw in ("measurable", "achievement", "accomplishment")):
            if a.strip():
                achievements_parts.append(a.strip())
    _add_section("Achievements", "\n".join(achievements_parts))

    # -- Certifications & Projects ---------------------------------------------
    certs_answer = ""
    for q, a in answers.items():
        if "certification" in q.lower() or "project" in q.lower() or "award" in q.lower():
            if a.strip():
                certs_answer = a.strip()
            break

    certs_body = _extract_section(original_text, "Certifications") or ""
    projects_body = _extract_section(original_text, "Projects") or ""
    combined = "\n".join(filter(None, [certs_body, projects_body, certs_answer]))
    _add_section("Certifications & Projects", combined)

    # -- Write PDF -------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdf.output(output_path)
    return os.path.abspath(output_path)

# ---------------------------------------------------------------------------
# 3. Automated Cover Letter Generator
# ---------------------------------------------------------------------------

def generate_cover_letter_pdf(
    original_text: str,
    skills: list[str],
    job_title: str,
    company: str,
    output_path: str = "cover_letter.pdf"
) -> str:
    """Generate a tailored Cover Letter PDF for a specific job match."""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    
    BLACK = (0, 0, 0)
    name = _safe_text(_extract_name(original_text))
    
    # Try to pull email and phone
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", original_text)
    phone_match = re.search(r"[\+]?[\d\s\-().]{7,15}", original_text)
    email = email_match.group(0) if email_match else "your.email@example.com"
    phone = phone_match.group(0).strip() if phone_match else "(555) 555-5555"

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 10, name, new_x="LMARGIN", new_y="NEXT", align="L")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, _safe_text(email), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 5, _safe_text(phone), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(10)
    
    import datetime
    today = datetime.datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 5, today, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, _safe_text(f"Hiring Manager"), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 5, _safe_text(company), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(10)
    
    # Body
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 5, "Dear Hiring Manager,", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(5)
    
    p1 = f"I am writing to express my strong interest in the {job_title} position at {company}. With my background in technology and a proven track record of delivering results, I am confident in my ability to make an immediate impact on your team."
    pdf.multi_cell(0, 6, _safe_text(p1))
    pdf.ln(5)
    
    top_skills = ", ".join(skills[:5]) if skills else "software development, problem-solving, and continuous learning"
    p2 = f"Throughout my career, I have developed expertise in {top_skills}. My experience aligns closely with the responsibilities of the {job_title} role, particularly my ability to adapt to new technologies and drive successful project outcomes."
    pdf.multi_cell(0, 6, _safe_text(p2))
    pdf.ln(5)
    
    p3 = f"What draws me to {company} is your commitment to innovation and excellence. I am excited about the opportunity to bring my technical skills and collaborative mindset to your engineering organization, and to contribute to the high-quality products your team builds."
    pdf.multi_cell(0, 6, _safe_text(p3))
    pdf.ln(5)
    
    p4 = "I would welcome the opportunity to discuss how my experience and vision align with the goals of your team. Thank you for considering my application. I have attached my resume for your review and look forward to the possibility of speaking with you soon."
    pdf.multi_cell(0, 6, _safe_text(p4))
    pdf.ln(10)
    
    pdf.cell(0, 6, "Sincerely,", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, name, new_x="LMARGIN", new_y="NEXT", align="L")
    
    # Write PDF
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdf.output(output_path)
    return os.path.abspath(output_path)
