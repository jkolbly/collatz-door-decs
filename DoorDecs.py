import math
from typing import List
import CollatzNames
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

class SpiralType:
  ARCHIMEDES = 1
  LOGARITHMIC = 2

class Settings:
  scale = 5 # A factor by which the image is scaled

  resolution = 150 * scale # DPI when converting to PDF
  page_size = (8.5, 11) # In inches
  pdf_margin = (0.5, 0.5) # In inches
  pdf_adaptive_scale = True
  pdf_target_width = 3.5 # Target door dec height, in inches
  pdf_target_height = 4.75 # In inches
  pdf_padding = 0.5 # Padding on all sides of each door dec, in inches

  initial_image_size = (2000 * scale, 5000 * scale)
  centerline = initial_image_size[0] // 2
  bg_color = (255, 255, 255, 255)

  show_title = True
  title_font_size = 60 * scale
  title_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=title_font_size)
  title_font_color = (0, 0, 0, 255)
  title_text = "Collatz Conjecture!"
  
  letter_padding = 0 * scale
  letter_font_size = 120 * scale
  letter_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=letter_font_size)
  letter_font_color = (0, 0, 0, 255)
  letter_spacing = 130 * scale # Distance between the centers of letters
  letter_circle_color = (56, 182, 255, 255)
  letter_circle_radius = 60 * scale

  arrow_tip_angle = 28 * math.pi / 180 # Angle between an arrow shaft and one half of the tip
  arrow_shaft_overlap = 2 * scale # Distance that arrow shaft overlaps with arrow tip, to avoid blank pixels due to rounding errors

  conversion_arrow_padding = 20 * scale # Padding between bottom of name and the top set of arrows
  conversion_arrowhead_size = 15 * scale # Size of the arrowhead for the top set of arrows, measured in length of each half of tip
  conversion_arrow_shaft_width = 5 * scale # Line width, including shaft and tip
  conversion_arrow_length = 40 * scale
  conversion_arrow_color = (100, 100, 100, 255)

  factor_padding = 0 * scale # Padding between the bottom of conversion arrows and the top (ascender) of factors
  factor_font_size = 80 * scale
  factor_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=factor_font_size)
  factor_font_color = (255, 145, 77, 255)

  x_font_size = 80 * scale
  x_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=x_font_size)
  x_font_color = (0, 0, 0, 255)
  x_centered = True # True if x's should be centered exactly between letter bboxes

  product_padding = 0 * scale
  product_font_size = 80 * scale
  product_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=product_font_size)
  product_font_color = (255, 0, 0, 255)
  equals_font_size = 80 * scale
  equals_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=equals_font_size)
  equals_font_color = (0, 0, 0, 255)

  show_explanation = True
  explanation_text_padding = 20 * scale
  explanation_font_size = 30 * scale
  explanation_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=explanation_font_size)
  explanation_font_color = (0, 0, 0, 255)
  explanation_text = "Even: ÷2\nOdd: x3+1"

  show_steps = True # Only appears if show_explanation is also True
  steps_padding = 0 * scale
  steps_font_size = 30 * scale
  steps_font = ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=steps_font_size)
  steps_font_color = (0, 150, 0, 255)
  steps_text_font_color = (0, 0, 0, 255)
  steps_text = " steps!"

  collatz_canvas_size = (2000 * scale, 2000 * scale)
  collatz_canvas_center = (collatz_canvas_size[0] // 2, collatz_canvas_size[1] // 2)
  collatz_padding = 10 * scale
  collatz_min_font_size = 12 * scale
  collatz_max_font_size = 20 * scale
  collatz_text_target_length = 35 * scale
  collatz_fonts = {size: ImageFont.truetype("./fonts/Work_Sans/WorkSans-VariableFont_wght.ttf", size=size) for size in range(collatz_min_font_size, collatz_max_font_size + 1)}
  collatz_start_color = product_font_color
  collatz_font_color = (0, 0, 0, 255)
  collatz_one_color = (0, 0, 255, 255) # The color of the 1 at the end of the Collatz sequence
  collatz_one_font_size = 30 * scale
  collatz_one_font = ImageFont.truetype("./fonts/Work_Sans/static/WorkSans-Bold.ttf", size=collatz_one_font_size)
  collatz_one_centered = True
  collatz_match_start_font = True # Whether the first number in the spiral should be colored product_font_color
  collatz_spiral_type = SpiralType.ARCHIMEDES # One of SpiralType.ARCHIMEDES or SpiralType.LOGARITHMIC
  collatz_spiral_slope = 6.5 * scale # For SpiralType.ARCHIMEDES, spiral is given by r = theta * collatz_spiral_slope
  collatz_spiral_a = 100 * scale # For SpiralType.LOGARITHMIC, spiral is given by r = collatz_spiral_a * e ^ ( collatz_spiral_k * theta )
  collatz_spiral_k = 0.0175 * scale
  collatz_spiral_deadzone = 0 # The deadzone (in radians) from the start of the spiral in which no numbers will be placed
  collatz_spacing = 80 * scale # The distance between adjacent numbers in the Collatz spiral, measured in arclength along the spiral
  collatz_spacing_epsilon = 0.01 # The margin of error for collatz_spacing
  collatz_spiral_clockwise = True
  collatz_rotated_text = False # Rotate the Collatz sequence numbers to be in line with the spiral
  collatz_arrow_color = (150, 150, 150, 255)
  collatz_arrow_padding = 5 * scale # Padding between a Collatz number bbox and the start of an arrow, measured as distance along the arrow
  collatz_arrow_shaft_width = 4 * scale
  collatz_arrowhead_size = 10 * scale

  background_color = (200, 200, 200, 255)
  background_corner_rounding_radius = 50 * scale
  background_padding = 20 * scale # Distance between edge of background and content
  background_bottom_padding = 50 * scale

  transparency_color = (255, 255, 255, 255)

def generate_doordec(name: CollatzNames.CollatzName):
  im = Image.new("RGBA", Settings.initial_image_size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(im)

  # This is the vertical anchor indicating where the top of the next thing should be drawn.
  # This changes as things are drawn, so the order in which things are drawn is important (top down).
  anchor = Settings.background_padding

  if Settings.show_title:
    draw.text((Settings.centerline, anchor), Settings.title_text, font=Settings.title_font, anchor="ma", fill=Settings.title_font_color)
    anchor += Settings.title_font_size + Settings.letter_padding

  prev_anchor = anchor

  letter_pos = Settings.centerline - len(name.name) // 2 * Settings.letter_spacing
  if len(name.name) % 2 == 0:
    letter_pos += Settings.letter_spacing // 2
  letter_bboxes = []
  for i in range(len(name.char_seq)):
    letter = name.name[i]
    factor = name.char_seq[i]

    anchor = prev_anchor

    letter_bbox = draw.textbbox((letter_pos, anchor), letter.upper(), font=Settings.letter_font, anchor="ma")
    draw.circle(get_bbox_center(letter_bbox), Settings.letter_circle_radius, fill=Settings.letter_circle_color)
    draw.text((letter_pos, anchor), letter.upper(), font=Settings.letter_font, anchor="ma", fill=Settings.letter_font_color)

    anchor += Settings.letter_font_size // 2 + Settings.letter_circle_radius + Settings.conversion_arrow_padding

    draw_arrow(
      draw,
      (letter_pos, anchor),
      (letter_pos, anchor + Settings.conversion_arrow_length),
      Settings.conversion_arrow_shaft_width, 
      Settings.conversion_arrowhead_size,
      Settings.conversion_arrow_color
    )

    anchor += Settings.conversion_arrow_length + Settings.factor_padding

    draw.text((letter_pos, anchor), str(factor), font=Settings.factor_font, anchor="ma", fill=Settings.factor_font_color)
    letter_bboxes.append(draw.textbbox((letter_pos, anchor), str(factor), font=Settings.factor_font, anchor="ma"))

    if not Settings.x_centered and i < len(name.char_seq) - 1:
      draw.text((letter_pos + Settings.letter_spacing // 2, anchor), "x", font=Settings.x_font, anchor="ma", fill=Settings.x_font_color)

    letter_pos += Settings.letter_spacing

  if Settings.x_centered:
    for i in range(len(letter_bboxes) - 1):
      bbox1 = letter_bboxes[i]
      bbox2 = letter_bboxes[i + 1]
      center = (bbox1[2] + bbox2[0]) // 2
      draw.text((center, anchor), "x", font=Settings.x_font, anchor="ma", fill=Settings.x_font_color)

  anchor += Settings.x_font_size + Settings.product_padding

  draw.text((Settings.centerline, anchor), str(name.int), font=Settings.product_font, anchor="ma", fill=Settings.product_font_color)
  product_length = draw.textlength(str(name.int), font=Settings.product_font)
  draw.text((Settings.centerline - product_length // 2, anchor), "= ", font=Settings.equals_font, anchor="ra", fill=Settings.equals_font_color)

  if Settings.show_explanation:
    size_bbox = draw.textbbox(
      (0, 0),
      Settings.explanation_text,
      font=Settings.explanation_font,
    )
    draw.text(
      (Settings.centerline + product_length // 2 + Settings.explanation_text_padding, anchor + Settings.product_font_size // 2 - (size_bbox[3] - size_bbox[1]) // 2),
      Settings.explanation_text,
      font=Settings.explanation_font,
      anchor="la",
      fill=Settings.explanation_font_color
    )

    if Settings.show_steps:
      explanation_bbox = draw.textbbox(
        (Settings.centerline + product_length // 2 + Settings.explanation_text_padding, anchor + Settings.product_font_size // 2 - (size_bbox[3] - size_bbox[1]) // 2),
        Settings.explanation_text,
        font=Settings.explanation_font,
        anchor="la"
      )
      draw.text(
        (Settings.centerline + product_length // 2 + Settings.explanation_text_padding, explanation_bbox[3] + Settings.steps_padding),
        str(len(name.collatz_seq)),
        font=Settings.steps_font,
        anchor="la",
        fill=Settings.steps_font_color
      )
      steps_length = draw.textlength(
        str(len(name.collatz_seq)),
        font=Settings.steps_font
      )
      draw.text(
        (Settings.centerline + product_length // 2 + Settings.explanation_text_padding + steps_length, explanation_bbox[3] + Settings.steps_padding),
        Settings.steps_text,
        font=Settings.steps_font,
        anchor="la",
        fill=Settings.steps_text_font_color
      )

  anchor += Settings.product_font_size + Settings.collatz_padding

  top_part_bbox = im.getbbox() # The bounding box for everything above the Collatz spiral

  spiral = draw_collatz_sequence(name.collatz_seq)
  im.paste(spiral, (Settings.centerline - spiral.width // 2, anchor), spiral)

  spiral_bbox = (
    Settings.centerline - spiral.width // 2,
    anchor,
    Settings.centerline - spiral.width // 2 + spiral.width,
    anchor + spiral.height
  )

  full_im = Image.new("RGBA", im.size, (0, 0, 0, 0))
  full_draw = ImageDraw.Draw(full_im)

  full_draw.rounded_rectangle(
    (
      [x - Settings.background_padding for x in top_part_bbox[:2]],
      [top_part_bbox[2] + Settings.background_padding, top_part_bbox[3] + Settings.background_bottom_padding],
    ),
    Settings.background_corner_rounding_radius,
    fill=Settings.background_color
  )

  full_draw.circle(
    get_bbox_center(spiral_bbox),
    max(spiral.width // 2, spiral.height // 2) + Settings.background_padding,
    fill=Settings.background_color
  )

  full_im.paste(im, (0, 0), im)
  full_im = full_im.crop(full_im.getbbox())

  non_transparent = Image.new("RGBA", full_im.size, Settings.transparency_color)
  non_transparent.paste(full_im, (0, 0), full_im)
 
  return full_im

def draw_collatz_sequence(sequence: List[int]):
  im = Image.new("RGBA", Settings.collatz_canvas_size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(im)

  # Try placing all numbers to establish a theta offset so that the starting number appears at the top
  thetas = [Settings.collatz_spiral_deadzone]
  for num in sequence:
    thetas.append(get_theta2(thetas[-1]))
  rot_offset = 3 * math.pi / 2 - (thetas[-1] * (-1 if Settings.collatz_spiral_clockwise else 1))

  prev_bbox = None
  for i in range(len(sequence)):
    font_color = Settings.collatz_font_color
    if i == 0 and Settings.collatz_match_start_font:
      font_color = Settings.product_font_color
    elif i == len(sequence) - 1:
      font_color = Settings.collatz_one_color

    num = sequence[i]
    theta = thetas[-1-i]

    spiral_coords = get_spiral_pos(theta)
    rotated = (
      spiral_coords[0] * math.cos(rot_offset) - spiral_coords[1] * math.sin(rot_offset),
      spiral_coords[0] * math.sin(rot_offset) + spiral_coords[1] * math.cos(rot_offset)
    )
    final_coords = (
      rotated[0] + Settings.collatz_canvas_center[0],
      rotated[1] + Settings.collatz_canvas_center[1]
    )

    if Settings.collatz_one_centered and num == 1:
      final_coords = Settings.collatz_canvas_center

    font = Settings.collatz_one_font if num == 1 else get_collatz_font(draw, str(num))

    this_bbox = None
    if Settings.collatz_rotated_text:
      max_size = (Settings.collatz_max_font_size * len(str(num)), Settings.collatz_max_font_size * len(str(num)))
      text_im = Image.new("RGBA", max_size, (0, 0, 0, 0))
      text_draw = ImageDraw.Draw(text_im)
      text_draw.text((max_size[0] // 2, max_size[1] // 2), str(num), font=font, anchor="mm", fill=font_color)
      rotated = text_im.rotate((math.pi + theta - rot_offset) * 180 / math.pi)
      cropped = rotated.crop(rotated.getbbox())
      this_bbox = (int(final_coords[0] - cropped.width / 2), int(final_coords[1] - cropped.height / 2), int(final_coords[0] + cropped.width / 2), int(final_coords[1] + cropped.height / 2))
      im.paste(rotated, (int(final_coords[0] - cropped.width / 2), int(final_coords[1] - cropped.height / 2)), rotated)
    else:
      draw.text(final_coords, str(num), font=font, anchor="mm", fill=font_color)
      this_bbox = draw.textbbox(final_coords, str(num), font=font, anchor="mm")

    if prev_bbox is not None:
      draw_arrow(draw,
        get_bbox_arrow_point(prev_bbox, this_bbox),
        get_bbox_arrow_point(this_bbox, prev_bbox),
        Settings.collatz_arrow_shaft_width,
        Settings.collatz_arrowhead_size,
        Settings.collatz_arrow_color
      )

    prev_bbox = this_bbox

  return symmetric_crop(im)

# Like im.crop(im.getbbox()) but the same amount is cropped from the left and right
def symmetric_crop(im: Image.Image):
  original = im.size
  bbox = im.getbbox()
  d1 = bbox[0]
  d2 = original[0] - bbox[2]
  cut_width = min(d1, d2)
  symmetric_bbox = (
    cut_width,
    bbox[1],
    original[0] - cut_width,
    bbox[3]
  )
  return im.crop(symmetric_bbox)

def get_bbox_center(bbox: tuple[int, int, int, int]):
  return (
    (bbox[0] + bbox[2]) // 2,
    (bbox[1] + bbox[3]) // 2
  )

# Get the unique point intersecting bbox1 along the line segment connecting the centers of bbox1 and bbox2.
# Includes additional padding given by Settings.collatz_arrow_padding
def get_bbox_arrow_point(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]):
  x1, y1, x2, y2 = bbox1
  center1 = get_bbox_center(bbox1)
  center2 = get_bbox_center(bbox2)

  delta_x = center2[0] - center1[0]
  delta_y = center2[1] - center1[1]

  d = 0
  common_radical = math.sqrt(delta_x * delta_x + delta_y * delta_y)
  if delta_x == 0 or abs(delta_y / delta_x) > abs((y2 - y1) / (x2 - x1)):
    # Case 1/3
    d = abs((y2 - y1) / (2 * delta_y) * common_radical)
  else:
    # Case 2
    d = abs((x2 - x1) / (2 * delta_x) * common_radical)
  d += Settings.collatz_arrow_padding
  
  return (
    int(center1[0] + d * delta_x / common_radical),
    int(center1[1] + d * delta_y / common_radical)
  )

def get_collatz_font(draw: ImageDraw.ImageDraw, text: str):
  for size in range(Settings.collatz_min_font_size + 1, Settings.collatz_max_font_size + 1):
    if draw.textlength(text, font=Settings.collatz_fonts[size]) > Settings.collatz_text_target_length:
      return Settings.collatz_fonts[size - 1]
  return Settings.collatz_fonts[Settings.collatz_max_font_size]

# Get position for a given theta for a spiral centered at the origin (with no rotation offset)
def get_spiral_pos(theta: float):
  if Settings.collatz_spiral_type == SpiralType.ARCHIMEDES:
    clockwise_factor = -1 if Settings.collatz_spiral_clockwise else 1
    return (
      math.cos(theta * clockwise_factor) * Settings.collatz_spiral_slope * theta,
      math.sin(theta * clockwise_factor) * Settings.collatz_spiral_slope * theta,
    )
  elif Settings.collatz_spiral_type == SpiralType.LOGARITHMIC:
    r = Settings.collatz_spiral_a * math.exp(Settings.collatz_spiral_k * theta)
    clockwise_factor = -1 if Settings.collatz_spiral_clockwise else 1
    return (
      math.cos(theta * clockwise_factor) * r,
      math.sin(theta * clockwise_factor) * r
    )

# Get the arclength of the spiral from theta=0 to the given theta
def get_total_arclength(theta: float):
  if Settings.collatz_spiral_type == SpiralType.ARCHIMEDES:
    return Settings.collatz_spiral_slope / 2 * (theta * math.sqrt(1 + theta * theta) + math.log(theta + math.sqrt(1 + theta * theta)))
  elif Settings.collatz_spiral_type == SpiralType.LOGARITHMIC:
    k = Settings.collatz_spiral_k
    a = Settings.collatz_spiral_a
    return math.sqrt(k * k + 1) / k * a * math.exp(k * theta)

# Get the arclength along the Archimedes spiral from angle theta1 to theta2
def get_arclength(theta1: float, theta2: float):
  return get_total_arclength(theta2) - get_total_arclength(theta1)

# Find a given theta2 given a desired arclength along the spiral.
# Estimated using a binary search for SpiralType.ARCHIMEDES because analytic solutions are hard :(
# The actual arclength will be within Settings.collatz_spacing_epsilon of Settings.collatz_spacing.
def get_theta2(theta1: float):
  if Settings.collatz_spiral_type == SpiralType.ARCHIMEDES:
    b = Settings.collatz_spiral_slope
    l = Settings.collatz_spacing
    pi = math.pi

    lowerbound = theta1 / 2 + math.sqrt(theta1 * theta1 / 4 + l / b) # Estimate using a circle of the larger radius
    upperbound = l / (b * theta1) + theta1 if theta1 != 0 else lowerbound * Settings.collatz_spiral_slope # Estimate using a circle of the smaller radius

    average = 0
    average_arclength = 0

    while abs(average_arclength - l) >= Settings.collatz_spacing_epsilon:
      average = (lowerbound + upperbound) / 2
      average_arclength = get_arclength(theta1, average)
      if average_arclength > l:
        upperbound = average
      else:
        lowerbound = average

    return average
  elif Settings.collatz_spiral_type == SpiralType.LOGARITHMIC:
    a = Settings.collatz_spiral_a
    k = Settings.collatz_spiral_k
    l = Settings.collatz_spacing
    return math.log(k * l / (a * math.sqrt(k * k + 1)) + math.exp(k * theta1)) / k

def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], shaft_width: int, arrowhead_size: int, color:tuple[int, int, int, int]):
  arrow_angle = math.atan2(end[1] - start[1], end[0] - start[0])

  arrowhead_length = math.cos(Settings.arrow_tip_angle) * arrowhead_size
  real_end = (
    end[0] - math.cos(arrow_angle) * (arrowhead_length - Settings.arrow_shaft_overlap),
    end[1] - math.sin(arrow_angle) * (arrowhead_length - Settings.arrow_shaft_overlap)
  )
  draw.line([start, real_end], width=shaft_width, fill=color)
  
  arrowhead_points = [end]
  for tip_angle in (arrow_angle + Settings.arrow_tip_angle, arrow_angle - Settings.arrow_tip_angle):
    tip_endpoint = (end[0] - math.cos(tip_angle) * arrowhead_size, end[1] - math.sin(tip_angle) * arrowhead_size)
    arrowhead_points.append(tip_endpoint)
  draw.polygon(arrowhead_points, fill=color)

# Create multiple images that are pages of a PDF containing the given images
def compile_pdf(images: List[Image.Image]) -> List[Image.Image]:
  page_size = [int(dim * Settings.resolution) for dim in Settings.page_size]
  margin = [int(dim * Settings.resolution) for dim in Settings.pdf_margin]
  bottom_right = [
    page_size[0] - margin[0],
    page_size[1] - margin[1]
  ]
  usable_size = [
    bottom_right[0] - margin[0],
    bottom_right[1] - margin[1]
  ]
  scale_target = [
    Settings.resolution * Settings.pdf_target_width,
    Settings.resolution * Settings.pdf_target_height
  ]
  padding = int(Settings.pdf_padding * Settings.resolution)

  image_queue = images.copy()
  image_queue.reverse()

  pages = []
  current_page = None
  current_pos = None
  row_height = None

  with tqdm(total=len(image_queue), desc="Placing on PDF") as pbar:
    while image_queue:
      if current_page is None:
        current_page = Image.new("RGBA", page_size, (0, 0, 0, 0))
        current_pos = [margin[0], margin[1]]
        row_height = 0

      im = image_queue[-1]

      if Settings.pdf_adaptive_scale:
        scale_to = (scale_target[0], im.height * scale_target[0] / im.width) if im.width / im.height > scale_target[0] / scale_target[1] \
          else (im.width * scale_target[1] / im.height, scale_target[1])
        scale_to = [int(x) for x in scale_to]
        im = im.resize(scale_to)

      if im.width > usable_size[0] or im.height > usable_size[1]:
        raise Exception("Image too big to fit on PDF!")
      if current_pos[0] + im.width > bottom_right[0]:
        current_pos = [
          margin[0],
          current_pos[1] + row_height + padding
        ]
      if current_pos[1] + im.height > bottom_right[1]:
        pages.append(current_page)
        current_page = None
        continue

      current_page.paste(im, (current_pos[0], current_pos[1]))
      current_pos[0] += im.width + padding
      row_height = max(row_height, im.height)
      image_queue.pop()
      pbar.update()

  if current_page != None:
    pages.append(current_page)

  return pages

if __name__ == "__main__":
  names = CollatzNames.get_name_objects("names.csv")
  images = []
  for name in tqdm(names, desc="Generating door decs"):
    im = generate_doordec(name)
    im.save(f"./out/{name.name}.png")
    images.append(im)
  pages = compile_pdf(images)
  print("Saving PDF...")
  pages[0].save("./out.pdf", save_all=True, append_images=pages[1:], resolution=Settings.resolution)
  print("Saved!")