# Conformité RGPD — Projet Electio-Analytics

## Contexte

Ce document formalise le cadre juridique et les mesures de protection des données appliquées dans le cadre du projet **ETL Elections IDF 2022** réalisé pour Electio-Analytics (MSPR TPRE813 — Bloc 3 Big Data & BI).

---

## 1. Nature des données utilisées

| Source | Type de données | Niveau d'agrégation | Données personnelles ? |
|--------|----------------|---------------------|------------------------|
| Ministère de l'Intérieur — Résultats électoraux | Votes exprimés par commune | Commune (≥ 1 268 unités) | **Non** — statistiques agrégées |
| INSEE — Recensement de la Population (RP 2022) | Structure démographique, CSP, diplômes | Commune | **Non** — ratios anonymisés |
| INSEE — Filosofi 2017/2021 | Revenus, pauvreté, déciles | IRIS / Commune | **Non** — indicateurs médians |
| INSEE — Taux de chômage 2009–2022 | Taux d'activité | Commune / Zone d'emploi | **Non** — taux agrégés |
| Ministère de l'Intérieur — Crimes et délits | Faits enregistrés par catégorie | Département | **Non** — comptages par département |
| INSEE Sirene — Établissements actifs | Nombre d'entreprises | Commune | **Non** — comptages bruts |
| RNA — Répertoire National des Associations | Nombre et type d'associations | Code postal | **Non** — comptages et ratios |

**Conclusion : aucune donnée personnelle nominative n'est traitée.** Toutes les données sont agrégées au niveau communal ou supérieur, ce qui rend toute ré-identification individuelle techniquement impossible.

---

## 2. Base légale de traitement (RGPD)

Le traitement s'appuie sur l'**Article 89 du RGPD** — *Garanties et dérogations applicables au traitement à des fins archivistiques dans l'intérêt public, à des fins de recherche scientifique ou historique ou à des fins statistiques*.

**Justifications :**
- Les données traitées sont publiques, issues de sources officielles de l'État français.
- Le projet poursuit une finalité de **recherche et d'analyse prospective** (prévision des tendances électorales).
- Aucune décision automatisée affectant des individus (article 22 RGPD) n'est produite : les modèles opèrent sur des agrégats communaux.
- Les résultats sont des **statistiques territoriales**, non des profils individuels.

---

## 3. Conformité des licences sources

| Source | Licence | Compatibilité réutilisation |
|--------|---------|----------------------------|
| data.gouv.fr (résultats électoraux) | Licence Ouverte Etalab v2.0 | ✅ Réutilisation libre, commerciale incluse |
| INSEE (RP, Filosofi, chômage) | Licence Ouverte Etalab v2.0 | ✅ Réutilisation libre sous mention de source |
| Ministère de l'Intérieur (crimes) | Licence Ouverte Etalab v2.0 | ✅ Réutilisation libre |
| Sirene (INSEE) | Licence Ouverte Etalab v2.0 | ✅ Réutilisation libre |
| RNA (data.gouv.fr) | Licence Ouverte Etalab v2.0 | ✅ Réutilisation libre |

La Licence Ouverte Etalab v2.0 est compatible avec la licence Creative Commons Attribution (CC BY). Elle autorise la réutilisation, la modification et la redistribution, y compris à des fins commerciales, sous réserve de mention de la source.

---

## 4. Absence de ré-identification

Le risque de ré-identification individuelle est évalué comme **nul** pour les raisons suivantes :

- **Granularité minimale : commune** — les plus petites unités traitées comptent plusieurs centaines d'habitants.
- **Indicateurs agrégés** — tous les chiffres sont des moyennes, médianes, ratios ou percentages calculés sur la population entière de la commune.
- **Pas de croisement avec identifiants** — aucune jointure avec des listes nominatives, adresses, numéros de téléphone ou toute donnée permettant d'identifier une personne physique.
- **Résultats électoraux** — le secret du vote est garanti constitutionnellement en France ; les données publiées par le Ministère sont des totaux par bureau de vote / commune.

---

## 5. Politique de rétention et de mise à jour

| Donnée | Fréquence de mise à jour | Politique de rétention |
|--------|--------------------------|------------------------|
| Résultats électoraux | Mise à jour après chaque scrutin | Conservation permanente (archive historique) |
| Données INSEE | Millésime annuel (publiées avec ~18 mois de décalage) | Remplacement par le millésime le plus récent |
| Données sécurité | Annuelle | Remplacement par l'année la plus récente |
| Données Sirene | Mensuelle (stock) | Remplacement à chaque recalcul du pipeline |
| Données RNA | Semestrielle | Remplacement à chaque recalcul |
| Artifacts ML (modèles .joblib) | À chaque exécution du pipeline | Conservation des 3 dernières versions |
| Logs d'audit | Continue | Conservation 90 jours, puis suppression |

---

## 6. Mesures de sécurité techniques

| Mesure | Détail |
|--------|--------|
| **Transport sécurisé** | HTTPS obligatoire via Nginx (TLS 1.2+) |
| **Isolation des containers** | Services Docker sur réseau bridge interne, aucun port DB exposé publiquement |
| **Contrôle d'accès DB** | 3 rôles PostgreSQL : `etl_writer` (écriture pipeline), `ds_reader` (lecture seule), `etl_admin` (admin restreint) |
| **Absence de secrets dans le code** | Mots de passe via variables d'environnement et fichier `.env` (non versionné) |
| **Pas de données personnelles stockées** | Aucun champ nom, prénom, adresse, email dans le schéma de données |
| **Données sources en lecture seule** | Les fichiers CSV sources ne sont jamais modifiés par le pipeline |

---

## 7. Droits des personnes

Les données traitées étant exclusivement des **statistiques agrégées et anonymisées** publiées par des organismes publics, les droits individuels RGPD (accès, rectification, effacement, portabilité) **ne s'appliquent pas** dans ce contexte — aucune donnée personnelle de personne physique identifiable n'est détenue.

En cas de question ou de demande liée à ce traitement, le point de contact est l'équipe projet (responsable technique du POC).

---

## 8. Registre de traitement (synthèse)

| Champ | Valeur |
|-------|--------|
| **Responsable de traitement** | Electio-Analytics (finalité) / Équipe projet ETL (mise en œuvre) |
| **Finalité** | Analyse prospective des tendances électorales — recherche & statistique |
| **Base légale** | Article 89 RGPD (finalité statistique / recherche) |
| **Catégories de données** | Données socio-économiques et électorales agrégées — **aucune donnée personnelle** |
| **Destinataires** | Équipe projet, directions métiers Electio-Analytics |
| **Durée de conservation** | Voir section 5 ci-dessus |
| **Transfert hors UE** | Aucun — infrastructure hébergée localement (Docker) |
| **Sous-traitants** | Aucun (traitement en environnement isolé) |

---

*Document généré dans le cadre de la MSPR TPRE813 — Big Data et Analyse de données — EPSI. Date : 2026-06-11.*
