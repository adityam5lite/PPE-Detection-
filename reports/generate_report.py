from fpdf import FPDF

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="PPE Safety Report", ln=True)

pdf.output("ppe_report.pdf")