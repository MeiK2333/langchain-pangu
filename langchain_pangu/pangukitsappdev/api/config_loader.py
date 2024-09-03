#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import importlib
import os
from typing import Dict, Any, Optional, Mapping, Tuple, Callable

from pydantic.v1 import BaseSettings, Extra
from pydantic.v1.env_settings import SettingsError, DotenvType, SettingsSourceCallable, EnvSettingsSource
from pydantic.v1.utils import deep_update

SDK_CONFIG_PATH_ENV = "SDK_CONFIG_PATH"
ENCRYPTED_PREFIX = "{Crypto."
SDK_CRYPTO_PATH_CONFIG = "sdk.crypto.implementation.path"

ConfigDecryptCallable = Callable[[str, str], str]


class SdkBaseSettings(BaseSettings):
    """
    Sdk的配置顶层类，覆盖了Pydantic的一些默认行为。通过在初始化的时候传递env_prefix参数来指定配置加载的key前缀，
    传递env_prefix_ident_fields参数，用于指定env_prefix对指定field生效，默认None对全部field生效
    """

    class Config:
        extra = Extra.ignore
        env_file = os.environ.get(SDK_CONFIG_PATH_ENV)

        @classmethod
        def customise_sources(
                cls,
                init_settings: SettingsSourceCallable,
                env_settings: SettingsSourceCallable,
                file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            env_file = env_settings.env_file if env_settings.env_file else os.environ.get(SDK_CONFIG_PATH_ENV)
            prop_settings = SdkEnvSettingsSource(env_file, env_settings.env_file_encoding,
                                                 env_settings.env_nested_delimiter, env_settings.env_prefix_len,
                                                 init_settings.init_kwargs.get("env_prefix"),
                                                 init_settings.init_kwargs.get("env_name_joint", "."),
                                                 init_settings.init_kwargs.get("env_prefix_ident_fields", None))
            return init_settings, prop_settings, file_secret_settings


class SdkEnvSettingsSource(EnvSettingsSource):
    """
    主干逻辑是__call__方法，结构上基本上和EnvSettingsSource保持一致。区别点：\n
    1. dotenv_vars配置文件的配置优先级高于环境变量.\n
    2. 增加env_prefix成员变量，用来实现匹配不同的前缀做配置加载\n
    3. 增加env_name_joint成员变量，默认使用"."连接env_prefix和env_name\n
    4. 增加config_decryptor类变量用来做配置的的解密，使用者可以自行配置。
    5. 增加env_prefix_ident_fields对指定元素生效，默认None
    """

    config_decryptor: ConfigDecryptCallable = None

    @classmethod
    def load_config_decryptor(cls, import_path: str) -> ConfigDecryptCallable:
        """从import_path加载config_decryptor
        加载自定义的config_decryptor。比如自定义的config_decryptor使用如下：\n
        from your.module import decrypt_func\n
        plain_text = decrypt_func(key_id, cipher)\n

        则等同使用如下方法加载：\n
        decrypt_func = load_config_decryptor("your.module.decrypt_func")\n
        plain_text = decrypt_func(key_id, cipher)\n

        该方法用来初始化类变量：SdkEnvSettingsSource.config_decryptor。只会初始化一次
        Args:
            import_path: 自定义config_decryptor的路径，基本格式，{path.to.module}.{function_name}

        Returns:
            一个可调用的方法对象

        """

        if cls.config_decryptor:
            return

        module_path, func_name = import_path.rsplit(".", 1)
        m = importlib.import_module(module_path)
        f = getattr(m, func_name)
        cls.config_decryptor = f

    __slots__ = (
        'env_file', 'env_file_encoding', 'env_nested_delimiter', 'env_prefix_len', 'env_prefix', 'env_name_joint',
        'env_prefix_ident_fields')

    def __init__(
            self,
            env_file: Optional[DotenvType],
            env_file_encoding: Optional[str],
            env_nested_delimiter: Optional[str] = None,
            env_prefix_len: int = 0,
            env_prefix: str = None,
            env_name_joint: str = ".",
            env_prefix_ident_fields: Optional[list] = None
    ):
        self.env_file: Optional[DotenvType] = env_file
        self.env_file_encoding: Optional[str] = env_file_encoding
        self.env_nested_delimiter: Optional[str] = env_nested_delimiter
        self.env_prefix_len: int = env_prefix_len
        # 在初始化的时候传入env_prefix，从环境变量中取值的时候考虑使用，默认是None
        self.env_prefix = env_prefix
        self.env_name_joint = env_name_joint
        self.env_prefix_ident_fields = env_prefix_ident_fields

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:  # noqa C901
        """
        Build environment variables suitable for passing to the Model.
        """
        d: Dict[str, Any] = {}

        if settings.__config__.case_sensitive:
            env_vars: Mapping[str, Optional[str]] = os.environ
        else:
            env_vars = {k.lower(): v for k, v in os.environ.items()}

        dotenv_vars = self._read_env_files(settings.__config__.case_sensitive)
        if dotenv_vars:
            """这里修改了，优先使用dotenv_vars里的配置，要求文件配置优先级高于环境变量配置，支持配置文件留空
            """
            dotenv_vars = {k: (dotenv_vars.get(k) if dotenv_vars.get(k) else None) for k in dotenv_vars.keys()}
            env_vars = {**env_vars, **dotenv_vars}

        # 加载load_config_decryptor
        sdk_crypto_path = env_vars.get(SDK_CRYPTO_PATH_CONFIG)
        if sdk_crypto_path:
            self.load_config_decryptor(sdk_crypto_path)

        for field in settings.__fields__.values():
            env_val: Optional[str] = None
            for env_name in field.field_info.extra['env_names']:
                need_prefix = self.env_prefix_ident_fields is None or env_name in self.env_prefix_ident_fields
                if self.env_prefix and need_prefix:
                    env_name = self.env_prefix.lower() + self.env_name_joint + env_name
                env_val = self.decrypt_if_need(env_vars.get(env_name))
                if env_val is not None:
                    break

            is_complex, allow_parse_failure = self.field_is_complex(field)
            if is_complex:
                if env_val is None:
                    # field is complex but no value found so far, try explode_env_vars
                    env_val_built = self.explode_env_vars(field, env_vars)
                    if env_val_built:
                        d[field.alias] = env_val_built
                else:
                    # field is complex and there's a value, decode that as JSON, then add explode_env_vars
                    try:
                        env_val = settings.__config__.parse_env_var(field.name, env_val)
                    except ValueError as e:
                        if not allow_parse_failure:
                            raise SettingsError(f'error parsing env var "{env_name}"') from e

                    if isinstance(env_val, dict):
                        d[field.alias] = deep_update(env_val, self.explode_env_vars(field, env_vars))
                    else:
                        d[field.alias] = env_val
            elif env_val is not None:
                # simplest case, field is not complex, we only need to add the value if it was found
                d[field.alias] = env_val

        return d

    @classmethod
    def decrypt_if_need(cls, config_value: str) -> str:
        if not cls.config_decryptor:
            return config_value

        if config_value and config_value.startswith(ENCRYPTED_PREFIX):
            key_id_end_idx = config_value.index("}")
            key_id = config_value[len(ENCRYPTED_PREFIX):key_id_end_idx]
            encrypted_value = config_value[key_id_end_idx + 1:]
            return cls.config_decryptor(key_id, encrypted_value)
        else:
            return config_value
