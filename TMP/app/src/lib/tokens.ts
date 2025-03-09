const ALPHABETS = {
  base32: "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567",
  urlsafe: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
  numeric: "0123456789",
} as const;

type TokenType = keyof typeof ALPHABETS;

interface TokenOptions {
  type: TokenType;
  length: number;
}

export const generateToken = async ({
  length,
  type,
}: TokenOptions): Promise<string> => {
  const alphabet = ALPHABETS[type];
  const randomBuffer = new Uint8Array(length);
  crypto.getRandomValues(randomBuffer);

  let token = "";
  for (let i = 0; i < randomBuffer.length; i++) {
    token += alphabet[randomBuffer[i] % alphabet.length];
  }

  return token;
};
