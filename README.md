# osu! Tournament OCR
## work in progress
## Usage
- Install tesseract
- (optional) create a virtual environment
- Install python dependencies `pip3 install -r requirements.txt`
- Update `pytesseract.pytesseract.tesseract_cmd` in `main.py` as needed
- Run `python3 main.py`
- Specify score bounds:
  1. Click top left corner of osu! tournament client team score box,
  2. Click bottom right corner of osu! tourney client's red team score box
  3. Click top left corner of osu! tourney client's blue team score box
  4. Click bottom right corner of osu! tourney client's blue team score box

