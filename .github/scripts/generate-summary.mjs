#!/usr/bin/env node
// Generates history/summary.json from history/*.yml and the status-branch
// commit history.  Replaces Upptime's built-in `readme` command because that
// command queries the GitHub Commits API without specifying `sha=status`,
// so it reads the default branch (main) and finds no Upptime data.

import { readFileSync, writeFileSync, readdirSync } from "fs";
import { join } from "path";

const OWNER = process.env.GITHUB_REPOSITORY?.split("/")[0] ?? "StarTrail-org";
const REPO = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "PixelRAG";
const BRANCH = "status";
const TOKEN = process.env.GH_PAT || process.env.GITHUB_TOKEN || "";
const CHECK_INTERVAL_MIN = 5;

// ---------------------------------------------------------------------------
// Minimal YAML helpers (only handles the flat key-value files we need)
// ---------------------------------------------------------------------------

function parseFlatYaml(text) {
  const obj = {};
  for (const line of text.split("\n")) {
    const m = line.match(/^([A-Za-z]\w*):\s+(.+)$/);
    if (!m) continue;
    let v = m[2].trim();
    if (/^\d+$/.test(v)) v = Number(v);
    obj[m[1]] = v;
  }
  return obj;
}

function parseUptimerc(text) {
  const sites = [];
  let current = null;
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    if (/^\s+-\s+name:\s+/.test(line)) {
      current = { name: line.replace(/^\s+-\s+name:\s+/, "").trim() };
      sites.push(current);
    } else if (current && /^\s+url:\s+/.test(line)) {
      current.url = line.replace(/^\s+url:\s+/, "").trim();
    }
  }
  return sites;
}

function slugify(name) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

// ---------------------------------------------------------------------------
// GitHub API – fetch commits on the status branch for a given history file
// ---------------------------------------------------------------------------

async function fetchAllCommits(slug) {
  const commits = [];
  for (let page = 1; page <= 10; page++) {
    const url =
      `https://api.github.com/repos/${OWNER}/${REPO}/commits` +
      `?sha=${BRANCH}&path=history/${slug}.yml&per_page=100&page=${page}`;
    const res = await fetch(url, {
      headers: {
        Accept: "application/vnd.github.v3+json",
        ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
      },
    });
    if (!res.ok) {
      console.error(`  API ${res.status} for ${slug} page ${page}`);
      break;
    }
    const data = await res.json();
    if (!data.length) break;
    commits.push(...data);
  }
  return commits;
}

// ---------------------------------------------------------------------------
// Parse Upptime commit messages
// ---------------------------------------------------------------------------

function parseResponseTime(msg) {
  const m = msg.match(/in\s+(\d+)\s+ms/);
  return m ? Number(m[1]) : null;
}

function commitStatus(msg) {
  const first = msg.split(" ")[0];
  if (first.includes("🟩")) return "up";
  if (first.includes("🟨")) return "degraded";
  if (first.includes("🟥")) return "down";
  return null; // non-Upptime commit
}

// ---------------------------------------------------------------------------
// Compute stats
// ---------------------------------------------------------------------------

function avg(arr) {
  return arr.length
    ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length)
    : 0;
}

function isWithin(dateStr, days) {
  return new Date(dateStr) > new Date(Date.now() - days * 86400000);
}

function computeUptime(entries, days) {
  const relevant = entries.filter((e) => isWithin(e.date, days));
  if (!relevant.length) return "100.00%";
  const up = relevant.filter((e) => e.status === "up").length;
  return (up / relevant.length * 100).toFixed(2) + "%";
}

function computeDailyMinutesDown(entries) {
  const daily = {};
  for (const e of entries) {
    if (e.status === "up") continue;
    const day = e.date.slice(0, 10);
    daily[day] = (daily[day] || 0) + CHECK_INTERVAL_MIN;
  }
  return daily;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const rcText = readFileSync(".upptimerc.yml", "utf8");
  const sites = parseUptimerc(rcText);
  console.log(`Found ${sites.length} sites in .upptimerc.yml`);

  const historyDir = join(process.cwd(), "history");
  const summary = [];

  for (const site of sites) {
    const slug = slugify(site.name);
    const histFile = join(historyDir, `${slug}.yml`);

    let hist;
    try {
      hist = parseFlatYaml(readFileSync(histFile, "utf8"));
    } catch {
      console.log(`  ${slug}: no history file, skipping`);
      continue;
    }

    console.log(`${slug}: status=${hist.status}, responseTime=${hist.responseTime}`);

    const commits = await fetchAllCommits(slug);
    const entries = commits
      .map((c) => ({
        date: c.commit.author.date,
        time: parseResponseTime(c.commit.message),
        status: commitStatus(c.commit.message),
      }))
      .filter((e) => e.status !== null);

    console.log(`  ${entries.length} Upptime commits (of ${commits.length} total)`);

    const dayTimes = entries.filter((e) => isWithin(e.date, 1) && e.time).map((e) => e.time);
    const weekTimes = entries.filter((e) => isWithin(e.date, 7) && e.time).map((e) => e.time);
    const monthTimes = entries.filter((e) => isWithin(e.date, 30) && e.time).map((e) => e.time);
    const yearTimes = entries.filter((e) => isWithin(e.date, 365) && e.time).map((e) => e.time);

    summary.push({
      name: site.name,
      url: hist.url || site.url,
      slug,
      status: hist.status || "down",
      uptime: computeUptime(entries, 365 * 10),
      uptimeDay: computeUptime(entries, 1),
      uptimeWeek: computeUptime(entries, 7),
      uptimeMonth: computeUptime(entries, 30),
      uptimeYear: computeUptime(entries, 365),
      time: hist.responseTime || avg(weekTimes),
      timeDay: avg(dayTimes),
      timeWeek: avg(weekTimes),
      timeMonth: avg(monthTimes),
      timeYear: avg(yearTimes),
      dailyMinutesDown: computeDailyMinutesDown(entries),
    });
  }

  const outPath = join(historyDir, "summary.json");
  writeFileSync(outPath, JSON.stringify(summary, null, 2) + "\n");
  console.log(`\nWrote ${outPath} (${summary.length} services)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
