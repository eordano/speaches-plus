use anyhow::Result;

pub fn env_arm_opt_in_eq_some_1_reread_every_call(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

pub fn env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name: &str) -> bool {
    std::env::var(name).ok().as_deref() != Some("0")
}

pub fn env_arm_present_at_all_reread_every_call(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

pub fn env_arm_opt_in_set_to_anything_but_0_reread_every_call(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v != "0")
}

#[macro_export]
macro_rules! env_arm_opt_in_eq_some_1_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var($name).ok().as_deref() == Some("1"))
    }};
}

#[macro_export]
macro_rules! env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var($name).ok().as_deref() != Some("0"))
    }};
}

#[macro_export]
macro_rules! env_arm_present_at_all_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var_os($name).is_some())
    }};
}

#[macro_export]
macro_rules! env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var($name).is_ok_and(|v| v != "0"))
    }};
}

pub fn env_arm_dependency_assert_enabled_arm_requires_prerequisite(
    arm_env: &str,
    arm_on: bool,
    prerequisite_env: &str,
    prerequisite_on: bool,
    why_the_arm_is_wrong_without_it: &str,
) -> Result<()> {
    anyhow::ensure!(
        !arm_on || prerequisite_on,
        "{arm_env}=1 needs {prerequisite_env}=1: {why_the_arm_is_wrong_without_it}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_arm_opt_in_reads_fresh_every_call() {
        let name = "NV_CFG_TEST_OPT_IN_REREAD";
        std::env::remove_var(name);
        assert!(!env_arm_opt_in_eq_some_1_reread_every_call(name));
        std::env::set_var(name, "1");
        assert!(env_arm_opt_in_eq_some_1_reread_every_call(name));
        std::env::set_var(name, "0");
        assert!(!env_arm_opt_in_eq_some_1_reread_every_call(name));
        std::env::remove_var(name);
    }

    #[test]
    fn env_arm_opt_out_defaults_on_and_only_literal_zero_disables() {
        let name = "NV_CFG_TEST_OPT_OUT_REREAD";
        std::env::remove_var(name);
        assert!(env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name));
        std::env::set_var(name, "0");
        assert!(!env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name));
        std::env::set_var(name, "off");
        assert!(env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name));
        std::env::remove_var(name);
    }

    #[test]
    fn env_arm_presence_triggers_on_any_value_including_0_and_empty() {
        let name = "NV_CFG_TEST_PRESENCE_REREAD";
        std::env::remove_var(name);
        assert!(!env_arm_present_at_all_reread_every_call(name));
        std::env::set_var(name, "0");
        assert!(env_arm_present_at_all_reread_every_call(name));
        std::env::set_var(name, "");
        assert!(env_arm_present_at_all_reread_every_call(name));
        std::env::remove_var(name);
        assert!(!env_arm_present_at_all_reread_every_call(name));
    }

    #[test]
    fn env_arm_set_to_anything_but_0_is_off_when_unset_or_literal_zero_only() {
        let name = "NV_CFG_TEST_SET_NOT0_REREAD";
        std::env::remove_var(name);
        assert!(!env_arm_opt_in_set_to_anything_but_0_reread_every_call(name));
        std::env::set_var(name, "0");
        assert!(!env_arm_opt_in_set_to_anything_but_0_reread_every_call(name));
        std::env::set_var(name, "off");
        assert!(env_arm_opt_in_set_to_anything_but_0_reread_every_call(name));
        std::env::set_var(name, "");
        assert!(env_arm_opt_in_set_to_anything_but_0_reread_every_call(name));
        std::env::remove_var(name);
    }

    #[test]
    fn env_arm_cached_macro_caches_per_call_site_so_the_call_sites_first_read_wins_over_later_env_changes(
    ) {
        let name = "NV_CFG_TEST_CACHED_FIRST_READ";
        fn one_call_site_like_a_family_wrapper_fn(name: &str) -> bool {
            env_arm_opt_in_eq_some_1_cached_first_read_wins!(name)
        }
        std::env::set_var(name, "1");
        assert!(one_call_site_like_a_family_wrapper_fn(name));
        std::env::set_var(name, "0");
        assert!(one_call_site_like_a_family_wrapper_fn(name));
        std::env::remove_var(name);
        assert!(!env_arm_opt_in_eq_some_1_cached_first_read_wins!(name));
    }

    #[test]
    fn env_arm_cached_opt_out_macro_defaults_on_and_first_read_wins() {
        let name = "NV_CFG_TEST_CACHED_OPT_OUT";
        fn one_call_site(name: &str) -> bool {
            env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins!(name)
        }
        std::env::set_var(name, "0");
        assert!(!one_call_site(name));
        std::env::remove_var(name);
        assert!(!one_call_site(name));
        assert!(env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins!(name));
    }

    #[test]
    fn env_arm_cached_presence_macro_first_read_wins_and_triggers_on_any_value() {
        let name = "NV_CFG_TEST_CACHED_PRESENCE";
        fn one_call_site(name: &str) -> bool {
            env_arm_present_at_all_cached_first_read_wins!(name)
        }
        std::env::set_var(name, "0");
        assert!(one_call_site(name));
        std::env::remove_var(name);
        assert!(one_call_site(name));
        assert!(!env_arm_present_at_all_cached_first_read_wins!(name));
    }

    #[test]
    fn env_arm_cached_set_not0_macro_first_read_wins() {
        let name = "NV_CFG_TEST_CACHED_SET_NOT0";
        fn one_call_site(name: &str) -> bool {
            env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins!(name)
        }
        std::env::set_var(name, "0");
        assert!(!one_call_site(name));
        std::env::set_var(name, "on");
        assert!(!one_call_site(name));
        assert!(env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins!(name));
        std::env::remove_var(name);
    }

    #[test]
    fn env_arm_dependency_assert_matches_the_shared_fold_needs_fused_chain_shape() {
        assert!(env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_X_CHILD",
            false,
            "NV_X_PARENT",
            false,
            "unused"
        )
        .is_ok());
        assert!(env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_X_CHILD",
            true,
            "NV_X_PARENT",
            true,
            "unused"
        )
        .is_ok());
        let err = env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_X_CHILD",
            true,
            "NV_X_PARENT",
            false,
            "the unfused path would scatter with weight 1.0",
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("NV_X_CHILD=1 needs NV_X_PARENT=1"));
        assert!(msg.contains("scatter with weight 1.0"));
    }
}
