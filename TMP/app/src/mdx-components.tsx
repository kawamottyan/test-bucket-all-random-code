import Link from "next/link";
import { ComponentPropsWithoutRef } from "react";

type HeadingProps = ComponentPropsWithoutRef<"h1">;
type ParagraphProps = ComponentPropsWithoutRef<"p">;
type ListProps = ComponentPropsWithoutRef<"ul">;
type ListItemProps = ComponentPropsWithoutRef<"li">;
type AnchorProps = ComponentPropsWithoutRef<"a">;

const components = {
  h1: (props: HeadingProps) => (
    <h1 className="mb-4 pt-12 text-4xl font-medium" {...props} />
  ),
  h2: (props: HeadingProps) => (
    <h2 className="mb-3 mt-8 text-3xl font-medium" {...props} />
  ),
  h3: (props: HeadingProps) => (
    <h3 className="mb-3 mt-8 text-2xl font-medium" {...props} />
  ),
  h4: (props: HeadingProps) => (
    <h4 className="mb-2 mt-6 text-xl font-medium" {...props} />
  ),
  p: (props: ParagraphProps) => (
    <p className="mb-4 text-base leading-relaxed" {...props} />
  ),
  ol: (props: ListProps) => (
    <ol className="mb-4 list-decimal space-y-2 pl-5 text-base" {...props} />
  ),
  ul: (props: ListProps) => (
    <ul className="mb-4 list-disc space-y-1 pl-5 text-base" {...props} />
  ),
  li: (props: ListItemProps) => <li className="pl-1" {...props} />,
  em: (props: ComponentPropsWithoutRef<"em">) => (
    <em className="font-medium" {...props} />
  ),
  a: ({ href, children, ...props }: AnchorProps) => {
    const className = "text-blue-500 hover:text-blue-700";
    if (href?.startsWith("/")) {
      return (
        <a href={href} className={className} {...props}>
          {children}
        </a>
      );
    }
    if (href?.startsWith("#")) {
      return (
        <Link href={href} className={className} {...props}>
          {children}
        </Link>
      );
    }
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={className}
        {...props}
      >
        {children}
      </a>
    );
  },
};

declare global {
  type MDXProvidedComponents = typeof components;
}

export function useMDXComponents(): MDXProvidedComponents {
  return components;
}
