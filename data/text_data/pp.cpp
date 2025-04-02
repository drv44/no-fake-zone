// #include <bits/stdc++.h>
// using namespace std;

// struct R {
//     string _name;               
//     vector<string> _cards; 
// };

// int value_of_rank(char r)
// {
//     if (r >= '2' && r <= '9')
//         return r - '0'; 
//     if (r == 'T')
//         return 10; 
//     if (r == 'J')
//         return 11; 
//     if (r == 'Q')
//         return 12; 
//     if (r == 'K')
//         return 13; 
//     if (r == 'A')
//         return 14;
//     return 0;      
// }

// R solution(vector<string> &cards) {
//     unordered_map<int, vector<string>> rr_map;
//     unordered_map<char, vector<string>> ss_map;
//     vector<int> uq;

//     for (const string &card : cards) {
//         int rank = value_of_rank(card[0]);
//         char suit = card[1];
//         rr_map[rank].push_back(card);
//         ss_map[suit].push_back(card);

//         if (find(uq.begin(), uq.end(), rank) == uq.end()) uq.push_back(rank);
        
//     }

    
//     sort(uq.rbegin(), uq.rend());

//     for (auto &entry : rr_map) if (entry.second.size() >= 3) return {"triple", {entry.second[0], entry.second[1], entry.second[2]}};

//     for (auto &entry : rr_map) if (entry.second.size() >= 2) return {"pair", {entry.second[0], entry.second[1]}};

//     for (size_t i = 0; i + 4 < uq.size(); i++)
//     {
//         if (uq[i] - 1 == uq[i + 1] &&
//             uq[i + 1] - 1 == uq[i + 2] &&
//             uq[i + 2] - 1 == uq[i + 3] &&
//             uq[i + 3] - 1 == uq[i + 4])
//         {
//             vector<string> selected;
//             for (int j = 0; j < 5; j++)
//                 selected.push_back(rr_map[uq[i + j]][0]);
//             return {"five in a row", selected};
//         }
//     }


//     for (auto &entry : ss_map) if (entry.second.size() >= 5) return {"suit", {entry.second.begin(), entry.second.begin() + 5}};

//     return {"single card", {rr_map[uq[0]][0]}};
// }

// // int main()
// // {
    
// //     vector<string> cards = {"2H", "4H", "7C", "9D", "TD", "KS"};

   
// //     Results result = solution(cards);

// //     cout << "{ \"set_name\": \"" << result.set_name << "\", \"selected_cards\": [";
// //     for (size_t i = 0; i < result.selected_cards.size(); i++)
// //     {
// //         if (i > 0)
// //             cout << ", ";
// //         cout << "\"" << result.selected_cards[i] << "\"";
// //     }
// //     cout << "] }" << endl;

// //     return 0;
// // }
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <climits>

using namespace std;

struct Results
{
    string set_name;
    vector<string> selected_cards;
};

struct Card
{
    int rank; // 0 (2) to 12 (A)
    int suit; // 3 (S), 2 (H), 1 (D), 0 (C)
    string original;
};

vector<Card> parse_cards(vector<string> &cards)
{
    vector<Card> res;
    for (string &s : cards)
    {
        int n = s.size();
        char suit_char = s.back();
        string rank_str = s.substr(0, n - 1);
        int rank;
        if (rank_str == "J")
        {
            rank = 9;
        }
        else if (rank_str == "Q")
        {
            rank = 10;
        }
        else if (rank_str == "K")
        {
            rank = 11;
        }
        else if (rank_str == "A")
        {
            rank = 12;
        }
        else if (rank_str == "10")
        {
            rank = 8;
        }
        else
        {
            rank = stoi(rank_str) - 2;
        }
        int suit;
        switch (suit_char)
        {
        case 'S':
            suit = 3;
            break;
        case 'H':
            suit = 2;
            break;
        case 'D':
            suit = 1;
            break;
        case 'C':
            suit = 0;
            break;
        }
        res.push_back({rank, suit, s});
    }
    return res;
}

Results check_full_house(const vector<Card> &parsed)
{
    unordered_map<int, int> rank_counts;
    for (const Card &c : parsed)
    {
        rank_counts[c.rank]++;
    }
    vector<int> possible_triple_ranks;
    for (const auto &p : rank_counts)
    {
        if (p.second >= 3)
        {
            possible_triple_ranks.push_back(p.first);
        }
    }
    sort(possible_triple_ranks.begin(), possible_triple_ranks.end(), greater<int>());

    for (int triple_rank : possible_triple_ranks)
    {
        vector<int> possible_pair_ranks;
        for (const auto &p : rank_counts)
        {
            if (p.first != triple_rank && p.second >= 2)
            {
                possible_pair_ranks.push_back(p.first);
            }
        }
        if (!possible_pair_ranks.empty())
        {
            sort(possible_pair_ranks.begin(), possible_pair_ranks.end(), greater<int>());
            int pair_rank = possible_pair_ranks[0];

            vector<string> selected;
            int count = 0;
            for (const Card &c : parsed)
            {
                if (c.rank == triple_rank && count < 3)
                {
                    selected.push_back(c.original);
                    count++;
                }
            }
            count = 0;
            for (const Card &c : parsed)
            {
                if (c.rank == pair_rank && count < 2)
                {
                    selected.push_back(c.original);
                    count++;
                }
            }
            return {"triple and pair", selected};
        }
    }
    return {"", {}};
}

Results check_flush(const vector<Card> &parsed)
{
    vector<int> suit_order = {3, 2, 1, 0};
    for (int suit : suit_order)
    {
        vector<Card> suited;
        for (const Card &c : parsed)
        {
            if (c.suit == suit)
            {
                suited.push_back(c);
            }
        }
        if (suited.size() >= 5)
        {
            sort(suited.begin(), suited.end(), [](const Card &a, const Card &b)
                 { return a.rank > b.rank; });
            vector<string> selected;
            for (int i = 0; i < 5; ++i)
            {
                selected.push_back(suited[i].original);
            }
            return {"suit", selected};
        }
    }
    return {"", {}};
}

Results check_straight(const vector<Card> &parsed)
{
    unordered_set<int> ranks;
    for (const Card &c : parsed)
    {
        ranks.insert(c.rank);
    }
    for (int start_rank = 12; start_rank >= 4; --start_rank)
    {
        bool is_straight = true;
        for (int i = 0; i < 5; ++i)
        {
            if (ranks.find(start_rank - i) == ranks.end())
            {
                is_straight = false;
                break;
            }
        }
        if (is_straight)
        {
            vector<string> selected;
            for (int i = 0; i < 5; ++i)
            {
                int target_rank = start_rank - i;
                for (const Card &c : parsed)
                {
                    if (c.rank == target_rank)
                    {
                        selected.push_back(c.original);
                        break;
                    }
                }
            }
            return {"five in a row", selected};
        }
    }
    return {"", {}};
}

Results check_triple(const vector<Card> &parsed)
{
    unordered_map<int, int> rank_counts;
    for (const Card &c : parsed)
    {
        rank_counts[c.rank]++;
    }
    int max_rank = -1;
    for (const auto &p : rank_counts)
    {
        if (p.second >= 3 && p.first > max_rank)
        {
            max_rank = p.first;
        }
    }
    if (max_rank != -1)
    {
        vector<string> selected;
        int count = 0;
        for (const Card &c : parsed)
        {
            if (c.rank == max_rank && count < 3)
            {
                selected.push_back(c.original);
                count++;
            }
        }
        return {"triple", selected};
    }
    return {"", {}};
}

Results check_pair(const vector<Card> &parsed)
{
    unordered_map<int, int> rank_counts;
    for (const Card &c : parsed)
    {
        rank_counts[c.rank]++;
    }
    int max_rank = -1;
    for (const auto &p : rank_counts)
    {
        if (p.second >= 2 && p.first > max_rank)
        {
            max_rank = p.first;
        }
    }
    if (max_rank != -1)
    {
        vector<string> selected;
        int count = 0;
        for (const Card &c : parsed)
        {
            if (c.rank == max_rank && count < 2)
            {
                selected.push_back(c.original);
                count++;
            }
        }
        return {"pair", selected};
    }
    return {"", {}};
}

Results check_single(const vector<Card> &parsed)
{
    int max_rank = -1;
    int max_suit = -1;
    string max_card;
    for (const Card &c : parsed)
    {
        if (c.rank > max_rank || (c.rank == max_rank && c.suit > max_suit))
        {
            max_rank = c.rank;
            max_suit = c.suit;
            max_card = c.original;
        }
    }
    return {"single card", {max_card}};
}

Results solution(vector<string> &cards)
{
    vector<Card> parsed = parse_cards(cards);

    Results result;

    result = check_full_house(parsed);
    if (!result.set_name.empty())
        return result;

    result = check_flush(parsed);
    if (!result.set_name.empty())
        return result;

    result = check_straight(parsed);
    if (!result.set_name.empty())
        return result;

    result = check_triple(parsed);
    if (!result.set_name.empty())
        return result;

    result = check_pair(parsed);
    if (!result.set_name.empty())
        return result;

    return check_single(parsed);
}
using namespace std;

int main()
    {

        vector<string> cards = {"2H", "4H", "7C", "9D", "TD", "KS"};

        Results result = solution(cards);

        cout << "{ \"set_name\": \"" << result.set_name << "\", \"selected_cards\": [";
        for (size_t i = 0; i < result.selected_cards.size(); i++)
        {
            if (i > 0)
                cout << ", ";
            cout << "\"" << result.selected_cards[i] << "\"";
        }
        cout << "] }" << endl;

        return 0;
    }