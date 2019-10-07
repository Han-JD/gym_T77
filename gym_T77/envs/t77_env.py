import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

# modules for game
import pygame
import random
import math

import logging
logger = logging.getLogger(__name__)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Set the height and width of the screen
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 240

# Number of bullets
N_BULLET = 50


class Bullet(pygame.sprite.Sprite):
    """ This class represents the bullet. """

    def __init__(self, start_x, start_y, dest_x, dest_y, velocity):
        """ Constructor.
        It takes in the starting x and y location.
        It also takes in the destination x and y position.
        """

        # Call the parent class (Sprite) constructor
        super().__init__()

        # Set up the image for the bullet
        self.image = pygame.Surface([2, 2])
        self.image.fill(YELLOW)

        self.rect = self.image.get_rect()

        # Move the bullet to our starting location
        self.rect.x = start_x
        self.rect.y = start_y

        # Because rect.x and rect.y are automatically converted
        # to integers, we need to create different variables that
        # store the location as floating point numbers. Integers
        # are not accurate enough for aiming.
        self.floating_point_x = start_x
        self.floating_point_y = start_y

        # Calculation the angle in radians between the start points
        # and end points. This is the angle the bullet will travel.
        x_diff = dest_x - start_x
        y_diff = dest_y - start_y
        angle = math.atan2(y_diff, x_diff)

        # Taking into account the angle, calculate our change_x
        # and change_y. Velocity is how fast the bullet travels.
        self.change_x = math.cos(angle) * velocity
        self.change_y = math.sin(angle) * velocity

    def update(self):
        """ Move the bullet. """

        # The floating point x and y hold our more accurate location.
        self.floating_point_y += self.change_y
        self.floating_point_x += self.change_x

        # The rect.x and rect.y are converted to integers.
        self.rect.y = int(self.floating_point_y)
        self.rect.x = int(self.floating_point_x)

        # If the bullet flies of the screen, get rid of it.
        if self.rect.x < 0 or self.rect.x > SCREEN_WIDTH or\
                self.rect.y < 0 or self.rect.y > SCREEN_HEIGHT:
            self.kill()


class Player(pygame.sprite.Sprite):
    """ Class to represent the player. """

    def __init__(self, color, screen_width, screen_height):
        """ Create the player image. """
        super().__init__()
        self.width = 6
        self.height = 6
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(color)
        self.rect = self.image.get_rect()

        # -- Attributes
        # Set start location
        self.rect.x = screen_width/2
        self.rect.y = screen_height/2
        # Set speed vector
        self.change_x = 0
        self.change_y = 0

    def change_speed(self, width, height):
        """ Change the speed of the player"""
        self.change_x += width
        self.change_y += height

    def update(self):
        """ Find a new position for the player"""
        self.rect.x += self.change_x
        self.rect.y += self.change_y
        if self.rect.x > SCREEN_WIDTH - self.width:
            self.rect.x = SCREEN_WIDTH - self.width
        elif self.rect.x < 0:
            self.rect.x = 0
        if self.rect.y > SCREEN_HEIGHT - self.height:
            self.rect.y = SCREEN_HEIGHT - self.height
        elif self.rect.y < 0:
            self.rect.y = 0


class T77Env(gym.Env):
    """
    Description:
        The player should avoid bullets to survive.

    Observation:
        Type: Box(51, 2)
        location of the bullets, location of the player
        Num     Observation         Min(int)    Max(int)
        0       player x, y           0           SCREEN_WIDTH
        1       bullet 01 x, y        0           SCREEN_HEIGHT
        2       bullet 02 x, y
        ...
        50      bullet 49 x, y
        51      bullet 50 x, y

        image of the view
         - next time

    Actions:
        Type: Discrete(5)
        There are 5 discrete deterministic actions:
        Num     Action
        0       not move
        1       move North
        2       move South
        3       move East
        4       move West
        issue: Human could push keys at the same time such as North and East, but not in this model.

    Rewards:
        +1 per each step(frame) while alive, -10000 for die.
        Maybe extra rewards for little moving or close passing, penalty for each moving

    Starting State:
        The player is located in the center of the screen.
        The 50 bullets are located in the edge of the screen.

    Episode Termination:
        When the player is hited a bullet
    """
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        self.compass = {
            0: (0, 0),  # 0 - None
            1: (0, -1),  # 1 - North
            2: (0, 1),  # 2 - South
            3: (1, 0),  # 3 - East
            4: (-1, 0)  # 4 - West
        }

        # Set a velocity of the bullet
        self.bullet_vel_max = 3
        self.bullet_vel_min = 0.7

        # Initialize Pygame
        pygame.init()

        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

        # Set the title of the window
        pygame.display.set_caption('BulletHell_ML')

        # Used to manage how fast the screen updates
        self.clock = pygame.time.Clock()

        # None, North, South, East, West
        self.action_space = spaces.Discrete(5)
        # x, y values for 1 player and 50 bullets
        self.observation_space = spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT),
                                            shape=(51, 2), dtype=np.int16)

        self.viewer = None
        # self.spaceViewer = Spaceview2D(screen_size=(320, 240), num_bullet=50)
        self.state = None

        # Loop until the user clicks the close button.
        self.game_over = False
        # This is a list of 'sprites.' Each block in the program is
        # added to this list. The list is managed by a class called 'Group.'
        self.bullet_list = pygame.sprite.Group()

        # This is a list of every sprite. All blocks and the player block as well.
        self.all_sprites_list = pygame.sprite.Group()

        # for Timer
        self.frame_count = 0
        self.frame_rate = 60
        self.start_time = 90

        # reward for printout
        self.total_reward = 0

        # Create a WHITE player block
        self.player = Player(WHITE, SCREEN_WIDTH, SCREEN_HEIGHT)

        # font
        self.small_font = pygame.font.SysFont("comicsansms", 12)

        self.seed()

        self.reset()

    # necessary?
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def text_objects(self, text, color, size):
        if size == "small":
            text_surface = self.small_font.render(text, True, color)
        elif size == "medium":
            text_surface = self.small_font.render(text, True, color)
        elif size == "large":
            text_surface = self.small_font.render(text, True, color)
        else:
            text_surface = self.small_font.render(text, True, color)

        return text_surface, text_surface.get_rect()

    def message_to_screen(self, msg, color, y_displace=0, size="small"):
        text_surf, text_rect = self.text_objects(msg, color, size)
        text_rect.center = (SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2) + y_displace
        self.screen.blit(text_surf, text_rect)

    def score(self, score):
        text = self.small_font.render("Score: " + str(score), True, WHITE)
        self.screen.blit(text, [0, 0])

    def random_bullet_generator(self):
        # Set a start location for the bullet
        rand = random.random()
        if rand < 0.25:
            start_x = 0
            start_y = random.randrange(SCREEN_HEIGHT)
        elif rand < 0.50:
            start_x = SCREEN_WIDTH
            start_y = random.randrange(SCREEN_HEIGHT)
        elif rand < 0.75:
            start_x = random.randrange(SCREEN_WIDTH)
            start_y = 0
        else:
            start_x = random.randrange(SCREEN_WIDTH)
            start_y = SCREEN_HEIGHT

        # Set a destination for the bullet
        bullet_type = random.random()
        if bullet_type < 0.7:
            dest_x = self.player.rect.x
            dest_y = self.player.rect.y

        else:
            dest_x = random.randrange(SCREEN_WIDTH)
            dest_y = random.randrange(SCREEN_HEIGHT)

        # Set a velocity of the bullet
        velocity = random.uniform(self.bullet_vel_min, self.bullet_vel_max)
        return start_x, start_y, dest_x, dest_y, velocity

    def step(self, action):
        """
        move agent (player)
        calculate a reward

        check game over

        :returns state, reward, game over
        """

        # make action into player
        action = self.compass.get(action)
        self.player.change_speed(action[0], action[1])
        """
        print("in step() before update")
        print("len(bullet_list): {}".format(len(self.bullet_list)))
        print("len(all_sprites_list): {}".format(len(self.all_sprites_list)))
        """
        # move all of the objects like player and bullets
        self.all_sprites_list.update()
        """
        print("in step() after update")
        print("len(bullet_list): {}".format(len(self.bullet_list)))
        print("len(all_sprites_list): {}".format(len(self.all_sprites_list)))
        """
        # Keep the number of bullets
        while len(self.bullet_list) < N_BULLET:
            start_x, start_y, dest_x, dest_y, velocity = self.random_bullet_generator()

            # This represents a bullet
            bullet = Bullet(start_x, start_y, dest_x, dest_y, velocity)

            # Add the block to the list of objects
            self.bullet_list.add(bullet)
            self.all_sprites_list.add(bullet)

        # set a state to return
        state = []

        for sprites in self.all_sprites_list:
            state.append((sprites.rect.x, sprites.rect.y))

        state = np.array(state, dtype=np.int16)
        assert len(state) == N_BULLET + 1, \
            "in step() len(state): {}, len(bullet_list): {}".format(len(state), len(self.bullet_list))

        # See if the player block has collided with anything.
        # when collide, a bullet is killed and the player is remained
        self.game_over = pygame.sprite.spritecollide(self.player, self.bullet_list, True)
        """
        if self.game_over:
            print("Game is OVER")
            print("len(bullet_list): {}".format(len(self.bullet_list)))
            print("len(all_sprites_list): {}".format(len(self.all_sprites_list)))
            print("has player? {}".format(self.all_sprites_list.has(self.player)))
        """
        # set a reward
        if self.game_over:
            reward = -100
            # print("Game OVER")
        else:
            reward = 1
        """
        # less moving is better!
        if action != 0:  # 0 = not moving
            reward -= 0.03
        """
        self.total_reward += reward

        done = bool(self.game_over)
        info = {}

        return state, reward, done, info

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        # print("reset() is called")
        # Loop until the user clicks the close button.
        self.game_over = False
        self.total_reward = 0
        # This is a list of 'sprites.' Each block in the program is
        # added to this list. The list is managed by a class called 'Group.'
        self.bullet_list = pygame.sprite.Group()

        # This is a list of every sprite. All blocks and the player block as well.
        self.all_sprites_list = pygame.sprite.Group()

        # Create a WHITE player block
        self.player = Player(WHITE, SCREEN_WIDTH, SCREEN_HEIGHT)

        self.all_sprites_list.add(self.player)

        # Generate bullets for first time
        for i in range(N_BULLET):
            # Set a start location for the bullet
            start_x, start_y, dest_x, dest_y, velocity = self.random_bullet_generator()

            # This represents a bullet
            # def __init__(self, start_x, start_y, dest_x, dest_y, velocity):
            bullet = Bullet(start_x, start_y, dest_x, dest_y, velocity)

            # Add the block to the list of objects
            self.bullet_list.add(bullet)
            self.all_sprites_list.add(bullet)
        """
        print("in reset()")
        print("len(bullet_list): {}".format(len(self.bullet_list)))
        print("len(all_sprites_list): {}".format(len(self.all_sprites_list)))
        """
        # set a state to return
        state = []

        for sprites in self.all_sprites_list:
            state.append((sprites.rect.x, sprites.rect.y))

        state = np.array(state, dtype=np.int16)
        assert len(state) == N_BULLET + 1, \
            "in reset() len(state): {}, len(bullet_list): {}, N_BULLET + 1: {}".format(len(state), len(self.bullet_list),
                                                                                      N_BULLET + 1)

        return state

    def render(self, mode='human'):
        """
        This function renders the current game state in the given mode.

        This command will display a popup window.
        Since it is written within a loop, an updated popup window will be rendered
        for every new action taken in each step.
        """
        # print("render() is called")

        # Clear the screen
        self.screen.fill(BLACK)

        # temp fixing.
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Only move and process game logic if the game isn't over.
        if not self.game_over:

            # Draw all the spites
            self.all_sprites_list.draw(self.screen)

            # ________Timer_Start________
            # --- Timer going up ---
            # Calculate total seconds
            total_seconds = self.frame_count // self.frame_rate

            # Divide by 60 to get total minutes
            minutes = total_seconds // 60

            # Use modulus (remainder) to get seconds
            seconds = total_seconds % 60

            '''
            # Use python string formatting to format in leading zeros
            output_string = "Time: {0:02}:{1:02}".format(minutes, seconds)

            # Blit to the screen
            text = font.render(output_string, True, WHITE)
            screen.blit(text, [0, 0])
            '''

            # Use python string formatting to format in leading zeros
            output_string = "total_reward: {}".format(self.total_reward)

            # Blit to the screen
            text = self.small_font.render(output_string, True, WHITE)
            self.screen.blit(text, [0, 0])

            # --- Timer going down ---
            # --- Timer going up ---
            # Calculate total seconds
            total_seconds = self.start_time - (self.frame_count // self.frame_rate)
            if total_seconds < 0:
                total_seconds = 0

            # Divide by 60 to get total minutes
            minutes = total_seconds // 60

            # Use modulus (remainder) to get seconds
            seconds = total_seconds % 60

            # Blit to the screen
            # text = self.font.render(output_string, True, BLACK)

            # self.screen.blit(text, [250, 280])

            # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
            self.frame_count += 1
            # Timer_end

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # Limit to 60 frames per second
        self.clock.tick(60)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pygame.quit()
        quit()




"""
if __name__ == '__main__':
    env = T77Env()
    total_reward = 0
    steps = 0
    state = env.reset()
    action = 0
    while True:
        # action = some_fuction(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        env.render()
        if done: break
    env.close()
"""