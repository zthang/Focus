import argparse
from task import WikiBioTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="medical")
    parser.add_argument("--model_path", type=str, default="/weights/llama/hf/")
    parser.add_argument("--only_keyword", action="store_true")
    parser.add_argument("--use_penalty", action="store_true")
    parser.add_argument("--add_type", action="store_true")
    parser.add_argument("--use_entropy", action="store_true")
    parser.add_argument("--use_idf", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--rho", type=float, default=0.01)
    args = parser.parse_args()
    if args.task == "wikibio":
        concepts = [
            f"michael savage",
            f"brian hughes"]
        outputs = [
            '''Michael Alan Weiner (born March 31, 1942), better known by his professional name Michael Savage, is an American radio host, author, activist, nutritionist, and conservative political commentator. He is the host of The Savage Nation, a nationally syndicated talk show that aired on Talk Radio Network across the United States until 2012, and in 2009 was the second most listened-to radio talk show in the country with an audience of over 20 million listeners on 400 stations across the United States. Since October 23, 2012, Michael Savage has been syndicated by Cumulus Media Networks. He holds master's degrees from the University of Hawaii in medical botany and medical anthropology, and a Ph.D. from the University of California, Berkeley in nutritional ethnomedicine. As Michael Weiner, he has written books on nutrition, herbal medicine, and homeopathy.''',
            '''Brian Hughes (born October 28, 1956) is a Canadian jazz guitarist, composer, and arranger. He has released several albums as a leader, and has performed and recorded with a variety of jazz, pop, and world music artists, including Kenny Rankin, Gino Vannelli, Joni Mitchell, Chaka Khan, and the Canadian Brass.\n\nHughes was born in Toronto, Ontario, Canada. He began playing guitar at the age of nine, and was influenced by jazz guitarists such as Wes Montgomery, Joe Pass, and George Benson. He studied music at York University in Toronto, and later at the Berklee College of Music in Boston.\n\nHughes has released several albums as a leader, including his debut album, "First Flight" (1985), and "One 2 One" (1995). He has also released several albums with his group, the Brian Hughes Quartet, including "Live at the Senator" (1997) and "Live at the Montreal Bistro" (1999). He has also released several albums with his world music group, the Brian Hughes/Michael O''']
        t = WikiBioTask(args)
        t.evaluate(concept=concepts[0], response=outputs[0], max_score=30.)
        t.evaluate(concept=concepts[1], response=outputs[1], max_score=30.)
