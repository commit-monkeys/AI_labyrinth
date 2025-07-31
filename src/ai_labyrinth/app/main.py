import pygame
import sys

# Exemple de matrice (1 = mur, 0 = chemin)
labyrinthe = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 3],
    [1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 2, 1, 1, 1, 1, 1, 1]
]

# Paramètres du rendu
TAILLE_CASE: int = 40
NB_LIGNES: int = len(labyrinthe)
NB_COLONNES: int = len(labyrinthe[0])
LARGEUR: int = NB_COLONNES * TAILLE_CASE
HAUTEUR: int = NB_LIGNES * TAILLE_CASE

# Couleurs
NOIR: tuple[int, int, int] = (0, 0, 0)
BLANC: tuple[int, int, int] = (255, 255, 255)
VERT: tuple[int, int, int] = (0, 255, 0)
ROUGE: tuple[int, int, int] = (255, 0, 0)

COULEURS = {
    1: NOIR,
    2: VERT,
    3: ROUGE,
}

def dessiner_labyrinthe(fenetre: pygame.Surface, matrice: list[list[int]]) -> None:
    for y, ligne in enumerate(matrice):
        for x, case in enumerate(ligne):
            couleur = COULEURS.get(case, BLANC)
            pygame.draw.rect(
                fenetre,
                couleur,
                pygame.Rect(x * TAILLE_CASE, y * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE)
            )

def main() -> None:
    pygame.init()
    fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption("Labyrinthe")

    clock = pygame.time.Clock()
    running = True

    while running:
        fenetre.fill(BLANC)
        dessiner_labyrinthe(fenetre, labyrinthe)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


