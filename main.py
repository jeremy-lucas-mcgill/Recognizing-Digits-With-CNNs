import pygame
import numpy as np
import torch
import torch.nn.functional as F
from cnn_pytorch import pytorchCNN

pygame.init()

#constants
width = 1000
height = 800
BLACK = (0,0,0)
WHITE = (255, 255, 255)
GRAY = (169,169,169)
RED = (227, 80, 75)
LIGHT_GRAY = (220,220,220)
grid_start = [50,50]
grid_end = [554,554]
grid_size = [28,28]
cell_width = (grid_end[0] - grid_start[0])/grid_size[0]
cell_height = (grid_end[1] - grid_start[1])/grid_size[1]
num_classes = 10
selected_class = np.zeros(num_classes)
font = pygame.font.Font(None, 36)
class_title_center = [650,50]
class_title_size = [250,50]
submit_button_center = [175, 600]
submit_button_size = [250,50]
clear_button_center = [175, 675]
clear_button_size = [250,50]
grid = np.zeros((grid_size[0],grid_size[1]))
print(cell_width, cell_height)
screen = pygame.display.set_mode((width, height))
#pytorch model initialization
model = pytorchCNN()
model_name = "pytorchCNN.pth"
model.load_state_dict(torch.load(model_name))
model.eval()

#draw interface
def drawDisplay(filled_cells, selected_class):
    #draw grid
    #gridlines
    for i in range(grid_size[0] + 1):
        line_start = [grid_start[0] + i * cell_width, grid_start[1]]
        line_end = [grid_start[0] + i * cell_width, grid_end[1]]
        pygame.draw.line(screen, BLACK,line_start, line_end, 1)
    for j in range(grid_size[1] + 1):
        line_start = [grid_start[0], grid_start[1] + j * cell_height]
        line_end = [grid_end[0], grid_start[1] + j * cell_height]
        pygame.draw.line(screen, BLACK,line_start, line_end, 1)
    #fill in corresponding cells
    for i in range(filled_cells.shape[0]):
        for j in range(filled_cells.shape[1]):
                value = (1 - filled_cells[i][j]) * 255
                rect = pygame.Rect(grid_start[0] + i * cell_width + 1, grid_start[1] + j * cell_height + 1, cell_width - 1, cell_height - 1)
                pygame.draw.rect(screen, (value,value, value), rect)
    # Draw class title
    rect = pygame.Rect(class_title_center[0], class_title_center[1], class_title_size[0], class_title_size[1])
    pygame.draw.rect(screen, GRAY, rect)
    text = font.render("CLASS", True, BLACK)
    text_rect = text.get_rect(center=rect.center)
    screen.blit(text, text_rect)
    #draw submit button
    rect = pygame.Rect(submit_button_center[0], submit_button_center[1], submit_button_size[0], submit_button_size[1])
    pygame.draw.rect(screen, GRAY, rect)
    pygame.draw.rect(screen, BLACK, rect, 3)
    text = font.render("SUBMIT", True, BLACK)
    text_rect = text.get_rect(center=rect.center)
    screen.blit(text, text_rect)
    #draw clear button
    rect = pygame.Rect(clear_button_center[0], clear_button_center[1], clear_button_size[0], clear_button_size[1])
    pygame.draw.rect(screen, RED, rect)
    pygame.draw.rect(screen, BLACK, rect, 3)
    text = font.render("CLEAR", True, BLACK)
    text_rect = text.get_rect(center=rect.center)
    screen.blit(text, text_rect)
    #draw class options
    rect = pygame.Rect(650,100, 250, 500)
    pygame.draw.rect(screen, LIGHT_GRAY, rect)
    for i in range(num_classes):
        #circle
        center = (775, 125 + i * 500 / num_classes)
        radius = 20
        value = (1-selected_class[i]) * 255
        pygame.draw.circle(screen, (value,value, value), center, radius)
        #class name
        text = font.render(str(i), True, BLACK)
        text_rect = pygame.Rect(725, 115 + i * 500 / num_classes, 20, 20)
        screen.blit(text, text_rect)
        #prediction values
        new_font = pygame.font.Font(None, 30)
        value = 0 if selected_class[i] < 0.5 else 255
        text = new_font.render(str(round(selected_class[i], 1)), True, (value, value, value))
        text_rect = pygame.Rect(760, 115 + i * 500 / num_classes, 20, 20)
        screen.blit(text, text_rect)

#mouse functions
def detectMouseOnGrid(mouse_clicks, mouse_pos):
    left, right, middle = mouse_clicks
    grid_pos_x = int((mouse_pos[0] - grid_start[0]) / cell_width)
    grid_pos_y = int((mouse_pos[1] - grid_start[1]) / cell_height)
    if (grid_pos_x < grid.shape[0] and grid_pos_x >= 0 and grid_pos_y < grid.shape[1] and grid_pos_y >= 0):
        if (left == 1):
            grid[grid_pos_x][grid_pos_y] = 1
        elif (right == 1):
            grid[grid_pos_x][grid_pos_y] = 0
def detectSubmitButtonClick(mouse_clicks, mouse_pos):
    left, right, middle = mouse_clicks
    if (left == 1):
        if submit_button_center[0] <= mouse_pos[0] <= submit_button_center[0] + submit_button_size[0] and submit_button_center[1] <= mouse_pos[1] <= submit_button_center[1] + submit_button_size[1]:
            return evaluateGrid(grid)
    return selected_class
def detectClearButtonClick(mouse_clicks, mouse_pos):
    left, right, middle = mouse_clicks
    if (left == 1):
        if clear_button_center[0] <= mouse_pos[0] <= clear_button_center[0] + clear_button_size[0] and clear_button_center[1] <= mouse_pos[1] <= clear_button_center[1] + clear_button_size[1]:
           return np.zeros((grid_size[0], grid_size[0])), np.zeros(num_classes)
    return grid, selected_class
#def evaluate model on grid
def evaluateGrid(input):
    input_list = np.transpose(input)
    input_array = np.array(input_list, dtype=np.float32)
    input_tensor = torch.tensor(input_array)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        softmax_output = F.softmax(output, dim = 1)
        sel = softmax_output.squeeze().cpu().numpy()
        return sel
#game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE)
    detectMouseOnGrid(pygame.mouse.get_pressed(), pygame.mouse.get_pos())
    selected_class = detectSubmitButtonClick(pygame.mouse.get_pressed(), pygame.mouse.get_pos())
    grid, selected_class = detectClearButtonClick(pygame.mouse.get_pressed(), pygame.mouse.get_pos())
    drawDisplay(filled_cells=grid, selected_class=selected_class)
    pygame.display.flip()
pygame.quit()