import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./api";

type Provider = "openai" | "gemini" | "deepseek" | "tavily";
type ProfileId = "full" | "lite";
type RoleMap = Record<string, string>;

export type KeyStatus = {
  configured: boolean;
  hint: string | null;
  variable?: string;
  source?: "system" | "env_file" | null;
};

export type EditableConfig = {
  profile: string;
  active?: ProfileId | null;
  keys: Record<Provider, KeyStatus>;
  roles: Array<{
    role: string;
    label: string;
    provider: string | null;
    model: string | null;
    effort: string | null;
    available: boolean;
    reason: string | null;
  }>;
  profile_source?: "file" | "env";
  profiles?: Record<string, RoleMap | { roles: RoleMap }>;
  seed?: Record<string, RoleMap | { roles: RoleMap }>;
  seed_dropped?: string[];
  editable?: boolean;
  env_error?: string | null;
};

type ModelOption = { id: string; image_input: boolean; efforts: string[] };
type ModelList = { loading: boolean; models: ModelOption[]; error: string | null };

const PROVIDER_LABELS: Record<Provider, string> = {
  openai: "OpenAI",
  gemini: "Gemini",
  deepseek: "DeepSeek",
  tavily: "Tavily",
};
const KEY_PROVIDERS: Provider[] = ["deepseek", "openai", "gemini", "tavily"];
const ROLES: Array<{ role: string; label: string; providers: Provider[] }> = [
  { role: "conversation", label: "对话", providers: ["openai", "deepseek"] },
  { role: "report", label: "写报告", providers: ["openai", "deepseek"] },
  { role: "research", label: "调查", providers: ["openai", "deepseek"] },
  { role: "research_max", label: "深度调查", providers: ["openai", "deepseek"] },
  { role: "vision", label: "看图", providers: ["openai", "deepseek"] },
  { role: "video", label: "视频与 PDF", providers: ["gemini"] },
  { role: "memory", label: "长期记忆", providers: ["openai"] },
];
const EFFORT_LABELS: Record<string, string> = {
  none: "不思考",
  minimal: "极少",
  low: "低",
  medium: "中",
  high: "高",
  xhigh: "极高",
  max: "最高",
};

function parseSpec(spec: string | undefined) {
  if (!spec) return null;
  const colon = spec.indexOf(":");
  if (colon < 0) return null;
  const provider = spec.slice(0, colon) as Provider;
  const rest = spec.slice(colon + 1);
  const at = rest.lastIndexOf("@");
  return at < 0
    ? { provider, model: rest, effort: null as string | null }
    : { provider, model: rest.slice(0, at), effort: rest.slice(at + 1) };
}

function formatSpec(provider: string, model: string, effort: string | null) {
  return `${provider}:${model}${effort ? `@${effort}` : ""}`;
}

function normalizeProfiles(
  source: EditableConfig["profiles"] | EditableConfig["seed"],
): Record<ProfileId, RoleMap> {
  const result: Record<ProfileId, RoleMap> = { full: {}, lite: {} };
  for (const id of ["full", "lite"] as ProfileId[]) {
    const value = source?.[id];
    if (!value) continue;
    const roles = "roles" in value && typeof value.roles === "object" ? value.roles : value;
    result[id] = { ...(roles as RoleMap) };
  }
  return result;
}

function effortsFor(role: string, provider: string, option: ModelOption | undefined) {
  if (role === "video" || role === "memory") return [];
  if (provider === "deepseek" && role !== "report") return [];
  return option?.efforts ?? [];
}

async function postConfig(path: string, body: unknown) {
  const response = await fetch(`${API_BASE}/api/config/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let payload: any = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  return { ok: response.ok, payload };
}

function errorText(payload: any, fallback = "操作失败") {
  if (payload && typeof payload.message === "string") return payload.message;
  return fallback;
}

export function ModelSettings({
  config,
  onConfigChanged,
}: {
  config: EditableConfig;
  onConfigChanged: (config: EditableConfig) => void;
}) {
  const active: ProfileId = config.active === "lite" ? "lite" : "full";
  const savedKey = JSON.stringify(
    normalizeProfiles(config.profile_source === "file" ? config.profiles : config.seed),
  );
  const saved = useMemo(() => JSON.parse(savedKey) as Record<ProfileId, RoleMap>, [savedKey]);
  const [view, setView] = useState<ProfileId>(active);
  const [draft, setDraft] = useState(saved);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [lists, setLists] = useState<Partial<Record<Provider, ModelList>>>({});
  const [busy, setBusy] = useState(false);
  const [roleNotes, setRoleNotes] = useState<Record<string, string>>({});
  const [roleErrors, setRoleErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [keyEditing, setKeyEditing] = useState<Provider | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [keyError, setKeyError] = useState<string | null>(null);

  useEffect(() => setDraft(saved), [saved]);
  useEffect(() => setView(active), [active]);

  const editable = config.editable === true;
  const roles = draft[view];
  const dirty = JSON.stringify(draft) !== JSON.stringify(saved);
  const runtimeRoles = new Map(config.roles.map((role) => [role.role, role]));

  const refresh = async () => {
    const response = await fetch(`${API_BASE}/api/config`);
    if (response.ok) onConfigChanged((await response.json()) as EditableConfig);
  };

  const loadModels = async (provider: Provider, force = false) => {
    if (!force && lists[provider] && !lists[provider]?.error) return;
    if (!config.keys[provider]?.configured) {
      setLists((current) => ({
        ...current,
        [provider]: { loading: false, models: [], error: "缺少 key" },
      }));
      return;
    }
    setLists((current) => ({ ...current, [provider]: { loading: true, models: [], error: null } }));
    const { ok, payload } = await postConfig("models", { provider });
    setLists((current) => ({
      ...current,
      [provider]: ok
        ? { loading: false, models: payload?.models ?? [], error: null }
        : { loading: false, models: [], error: errorText(payload, "无法获取模型列表") },
    }));
  };

  const setRole = (role: string, spec: string | null) => {
    setDraft((current) => {
      const next = { ...current, [view]: { ...current[view] } };
      if (spec) next[view][role] = spec;
      else delete next[view][role];
      return next;
    });
    setRoleErrors((current) => {
      const next = { ...current };
      delete next[`${view}.${role}`];
      return next;
    });
  };

  const toggleRole = (role: string) => {
    if (!editable) return;
    const next = expanded === role ? null : role;
    setExpanded(next);
    if (!next) return;
    const spec = parseSpec(roles[role]);
    const providers = ROLES.find((item) => item.role === role)?.providers ?? [];
    const provider = spec?.provider ?? providers[view === "lite" && providers.length > 1 ? 1 : 0];
    if (provider && role !== "memory") void loadModels(provider as Provider);
  };

  const save = async (): Promise<boolean> => {
    const incompleteRole = (["full", "lite"] as const)
      .flatMap((profile) =>
        Object.entries(draft[profile]).map(([role, value]) => ({ profile, role, value })),
      )
      .find(({ value }) => {
        const spec = parseSpec(value);
        return spec !== null && Boolean(spec.provider) && !spec.model.trim();
      });
    if (incompleteRole) {
      setFormError(null);
      setRoleNotes({});
      setRoleErrors({
        [`${incompleteRole.profile}.${incompleteRole.role}`]: "还没选模型",
      });
      setView(incompleteRole.profile);
      setExpanded(incompleteRole.role);
      return false;
    }
    setBusy(true);
    setFormError(null);
    setRoleErrors({});
    setRoleNotes({});
    const profiles = {
      full: { roles: draft.full },
      lite: { roles: draft.lite },
    };
    try {
      const { ok, payload } = await postConfig("profiles", { profiles });
      if (!ok) {
        if (Array.isArray(payload?.errors)) {
          const errors: Record<string, string> = {};
          for (const item of payload.errors) {
            const fallback = String(item.role).endsWith("_fallback");
            const base = String(item.role).replace(/_fallback$/, "");
            errors[`${item.profile}.${base}`] = fallback ? `备用模型：${item.message}` : item.message;
          }
          setRoleErrors(errors);
        } else {
          setFormError(errorText(payload, "保存失败"));
        }
        return false;
      }
      const notes: Record<string, string> = {};
      for (const item of payload?.untested ?? [])
        notes[`${item.profile}.${String(item.role).replace(/_fallback$/, "")}`] = item.reason
          ? `${item.reason}，未测试`
          : "未测试";
      setRoleNotes(notes);
      setExpanded(null);
      if (payload?.config) onConfigChanged(payload.config as EditableConfig);
      else await refresh();
      return true;
    } catch {
      setFormError("无法连接后端");
      return false;
    } finally {
      setBusy(false);
    }
  };

  const switchProfile = async () => {
    if (config.profile_source !== "file") {
      const conversation = parseSpec(draft[view].conversation);
      if (!conversation?.model || !config.keys[conversation.provider]?.configured) {
        setFormError(conversation?.model ? "对话模型缺少 key" : "未设置对话模型");
        return;
      }
      if (!(await save())) return;
    }
    setBusy(true);
    setFormError(null);
    try {
      const { ok, payload } = await postConfig("active", { active: view });
      if (!ok) {
        const first = Array.isArray(payload?.errors) ? payload.errors[0] : null;
        setFormError(first ? `对话模型：${first.message}` : errorText(payload, "切换失败"));
        return;
      }
      if (payload?.config) onConfigChanged(payload.config as EditableConfig);
      else await refresh();
    } catch {
      setFormError("无法连接后端");
    } finally {
      setBusy(false);
    }
  };

  const saveKey = async (provider: Provider) => {
    const value = keyInput.trim();
    setKeyInput("");
    if (!value) return;
    setBusy(true);
    setKeyError(null);
    try {
      const { ok, payload } = await postConfig("key", { provider, api_key: value });
      if (!ok) {
        setKeyError(errorText(payload, "保存失败"));
        return;
      }
      setKeyEditing(null);
      setLists((current) => ({ ...current, [provider]: undefined }));
      await refresh();
    } catch {
      setKeyError("无法连接后端");
    } finally {
      setBusy(false);
    }
  };

  const renderEditor = (role: string, providers: Provider[]) => {
    const spec = parseSpec(roles[role]);
    if (role === "memory") {
      const enabled = Boolean(spec);
      const memorySpec =
        roles.memory ??
        saved.full.memory ??
        `openai:${runtimeRoles.get("memory")?.model ?? "text-embedding-3-small"}`;
      return (
        <div className="model-editor">
          <div className="model-segment" role="radiogroup" aria-label="长期记忆">
            <button type="button" className={enabled ? "active" : ""} onClick={() => setRole(role, memorySpec)}>
              开
            </button>
            <button type="button" className={!enabled ? "active" : ""} onClick={() => setRole(role, null)}>
              关
            </button>
          </div>
        </div>
      );
    }
    const provider = (spec?.provider ?? providers[view === "lite" && providers.length > 1 ? 1 : 0]) as Provider;
    const list = lists[provider];
    const options = list?.models ?? [];
    const option = options.find((item) => item.id === spec?.model);
    const efforts = effortsFor(role, provider, option);
    const pick = (model: string, effort: string | null) => {
      if (!model) {
        setRole(role, `${provider}:`);
        return;
      }
      const nextOption = options.find((item) => item.id === model);
      const allowed = effortsFor(role, provider, nextOption);
      const nextEffort =
        allowed.length === 0
          ? provider === "deepseek" && role !== "report"
            ? "none"
            : null
          : effort && allowed.includes(effort)
            ? effort
            : provider === "openai"
              ? null
              : allowed[0];
      setRole(role, model ? formatSpec(provider, model, nextEffort) : null);
    };
    return (
      <div className="model-editor">
        {providers.length > 1 && (
          <div className="model-segment" role="radiogroup" aria-label="服务商">
            {providers.map((item) => (
              <button
                type="button"
                key={item}
                className={item === provider ? "active" : ""}
                onClick={() => {
                  if (item === provider) return;
                  setRole(role, `${item}:`);
                  void loadModels(item);
                }}
              >
                {PROVIDER_LABELS[item]}
              </button>
            ))}
          </div>
        )}
        {provider === "openai" ? (
          <>
            <input
              className="model-input"
              list={`models-${provider}`}
              value={spec?.model ?? ""}
              placeholder="模型名称"
              spellCheck={false}
              autoComplete="off"
              onChange={(event) => pick(event.target.value.trim(), spec?.effort ?? null)}
            />
            <datalist id={`models-${provider}`}>
              {options.map((item) => (
                <option key={item.id} value={item.id} />
              ))}
            </datalist>
          </>
        ) : (
          <select
            className="model-input"
            value={spec?.model ?? ""}
            onChange={(event) => pick(event.target.value, spec?.effort ?? null)}
          >
            <option value="" disabled>
              {list?.loading ? "读取中…" : "选择模型"}
            </option>
            {spec?.model && !option && <option value={spec.model}>{spec.model}</option>}
            {options.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}
                {role === "vision" && !item.image_input ? "（不能看图）" : ""}
              </option>
            ))}
          </select>
        )}
        {efforts.length > 1 && spec?.model && (
          <div className="model-segment" role="radiogroup" aria-label="思考">
            {(provider === "openai" ? [null, ...efforts] : efforts).map((effort) => (
              <button
                type="button"
                key={effort ?? "default"}
                className={(spec.effort ?? (provider === "openai" ? null : "none")) === effort ? "active" : ""}
                onClick={() => pick(spec.model, effort)}
              >
                {effort ? EFFORT_LABELS[effort] ?? effort : "默认"}
              </button>
            ))}
          </div>
        )}
        {list?.error && <span className="config-reason">{list.error}</span>}
        {role !== "conversation" && spec && (
          <button type="button" className="model-link" onClick={() => setRole(role, null)}>
            不使用
          </button>
        )}
      </div>
    );
  };

  return (
    <>
      <div className="model-segment model-profile-switch" role="tablist" aria-label="配置">
        {(["full", "lite"] as ProfileId[]).map((id) => (
          <button
            type="button"
            role="tab"
            aria-selected={view === id}
            key={id}
            className={view === id ? "active" : ""}
            onClick={() => {
              setView(id);
              setExpanded(null);
              setFormError(null);
            }}
          >
            {id === "full" ? "完整" : "便捷"}
            {active === id && <span className="model-active-dot" aria-label="使用中" />}
          </button>
        ))}
      </div>

      <div className="config-role-list">
        {ROLES.map(({ role, label, providers }) => {
          const spec = parseSpec(roles[role]);
          const runtime = view === active && !dirty ? runtimeRoles.get(role) : undefined;
          const key = `${view}.${role}`;
          const missingKey = spec && !config.keys[spec.provider]?.configured;
          const unavailable = runtime ? !runtime.available : !spec || Boolean(missingKey);
          const reason = roleErrors[key] ?? roleNotes[key] ?? (!spec ? null : runtime && !runtime.available ? runtime.reason : missingKey ? "缺少 key" : null);
          const modelLabel =
            role === "memory"
              ? spec
                ? "开"
                : "关"
              : spec?.model || (spec ? PROVIDER_LABELS[spec.provider] : "未配置");
          return (
            <div key={role} className="model-role">
              <button
                type="button"
                className={`config-role-row model-role-button ${unavailable ? "unavailable" : ""}`}
                aria-expanded={expanded === role}
                disabled={!editable}
                onClick={() => toggleRole(role)}
              >
                <span className={`config-status-dot ${!spec ? "unset" : unavailable ? "offline" : "online"}`} />
                <span className="config-role-label">{label}</span>
                <span className="config-role-model" title={spec?.model || (spec ? PROVIDER_LABELS[spec.provider] : "未配置")}>
                  {modelLabel}
                </span>
                {spec?.effort && spec.effort !== "none" && role !== "memory" ? (
                  <span className="config-effort">{EFFORT_LABELS[spec.effort] ?? spec.effort}</span>
                ) : (
                  <span />
                )}
                {reason && <span className={`config-reason ${roleErrors[key] ? "" : "muted"}`}>{reason}</span>}
              </button>
              {expanded === role && renderEditor(role, providers)}
            </div>
          );
        })}
      </div>

      {config.env_error && <div className="config-reason">{config.env_error}</div>}
      {!editable && !config.env_error && <div className="settings-empty">只能在本机修改配置</div>}

      {editable && dirty && (
        <div className="model-actions">
          {config.profile_source === "env" && (config.seed_dropped?.length ?? 0) > 0 && (
            <span className="config-reason muted">
              保存后这些设置不再生效：{config.seed_dropped!.join("、")}
            </span>
          )}
          <button type="button" className="model-link" disabled={busy} onClick={() => {
            setDraft(saved);
            setRoleErrors({});
            setExpanded(null);
          }}>
            还原
          </button>
          <button type="button" className="model-primary" disabled={busy} onClick={() => void save()}>
            {busy ? "测试中…" : "测试并保存"}
          </button>
        </div>
      )}
      {editable && !dirty && view !== active && (
        <div className="model-actions">
          <button type="button" className="model-primary" disabled={busy} onClick={() => void switchProfile()}>
            {busy ? "切换中…" : "切换到此配置"}
          </button>
        </div>
      )}
      {formError && <div className="config-reason">{formError}</div>}

      <div className="config-services">
        <span className="config-services-title">API Key</span>
        {KEY_PROVIDERS.map((provider) => {
          const key = config.keys[provider];
          if (!key) return null;
          const system = key.source === "system";
          return (
            <div key={provider} className="model-key">
              <div className="config-service-row">
                <span className="config-service-name">
                  <span className={`config-status-dot ${key.configured ? "online" : "offline"}`} />
                  {PROVIDER_LABELS[provider]}
                </span>
                <span className="model-key-side">
                  <span className={`config-service-detail ${key.configured ? "" : "unavailable"}`}>
                    {system ? "系统环境变量" : key.configured ? key.hint ?? "已配置" : "未配置"}
                  </span>
                  {editable && !system && keyEditing !== provider && (
                    <button
                      type="button"
                      className="model-link"
                      onClick={() => {
                        setKeyEditing(provider);
                        setKeyInput("");
                        setKeyError(null);
                      }}
                    >
                      {key.configured ? "替换" : "填写"}
                    </button>
                  )}
                </span>
              </div>
              {keyEditing === provider && (
                <form
                  className="model-key-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    void saveKey(provider);
                  }}
                >
                  <input
                    className="model-input"
                    type="password"
                    name={`mercy-${provider}-key`}
                    autoComplete="new-password"
                    spellCheck={false}
                    autoFocus
                    value={keyInput}
                    placeholder={`${PROVIDER_LABELS[provider]} key`}
                    onChange={(event) => setKeyInput(event.target.value)}
                  />
                  <button type="submit" className="model-primary" disabled={busy || !keyInput.trim()}>
                    {busy ? "验证中…" : "保存"}
                  </button>
                  <button
                    type="button"
                    className="model-link"
                    onClick={() => {
                      setKeyEditing(null);
                      setKeyInput("");
                      setKeyError(null);
                    }}
                  >
                    取消
                  </button>
                  {keyError && <span className="config-reason">{keyError}</span>}
                </form>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
