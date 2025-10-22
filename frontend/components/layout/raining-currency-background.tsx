"use client";

/**
 * Raining currency background effect with Divine Orbs and Mirror of Kalandra
 */

import { useEffect, useRef, useState } from "react";
import { useBackgroundEffect } from "@/lib/providers/background-effect-provider";

interface CurrencyParticle {
  x: number;
  y: number;
  speed: number;
  rotation: number;
  rotationSpeed: number;
  size: number;
  type: "divine" | "mirror";
  opacity: number;
}

interface CurrencyImages {
  divine: HTMLImageElement | null;
  mirror: HTMLImageElement | null;
}

export function RainingCurrencyBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { isEnabled } = useBackgroundEffect();
  const particlesRef = useRef<CurrencyParticle[]>([]);
  const animationFrameRef = useRef<number>();
  const [currencyImages, setCurrencyImages] = useState<CurrencyImages>({
    divine: null,
    mirror: null,
  });
  const [isMounted, setIsMounted] = useState(false);

  // Prevent hydration mismatch by only rendering on client
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Load currency images from local files
  useEffect(() => {
    // Use local PNG files stored in public/images
    const localUrls = {
      divine: "/images/divine-orb.png",
      mirror: "/images/mirror-of-kalandra.png",
    };

    console.log("🎨 Loading currency images from local files...");
    console.log("Divine URL:", localUrls.divine);
    console.log("Mirror URL:", localUrls.mirror);

    // Load Divine Orb image
    const divineImg = new Image();
    divineImg.onload = () => {
      console.log("✅ Divine Orb image loaded successfully");
      setCurrencyImages((prev) => ({ ...prev, divine: divineImg }));
    };
    divineImg.onerror = (e) => {
      console.error("❌ Failed to load Divine Orb image:", e);
      console.error("URL was:", localUrls.divine);
    };
    divineImg.src = localUrls.divine;

    // Load Mirror of Kalandra image
    const mirrorImg = new Image();
    mirrorImg.onload = () => {
      console.log("✅ Mirror of Kalandra image loaded successfully");
      setCurrencyImages((prev) => ({ ...prev, mirror: mirrorImg }));
    };
    mirrorImg.onerror = (e) => {
      console.error("❌ Failed to load Mirror image:", e);
      console.error("URL was:", localUrls.mirror);
    };
    mirrorImg.src = localUrls.mirror;
  }, []);

  useEffect(() => {
    if (!isEnabled) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      particlesRef.current = [];
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Initialize particles
    const particleCount = 20;
    if (particlesRef.current.length === 0) {
      for (let i = 0; i < particleCount; i++) {
        particlesRef.current.push(createParticle());
      }
    }

    function createParticle(): CurrencyParticle {
      return {
        x: Math.random() * canvas!.width,
        y: Math.random() * -canvas!.height,
        speed: Math.random() * 2 + 1,
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.05,
        size: Math.random() * 30 + 20,
        type: Math.random() > 0.5 ? "divine" : "mirror",
        opacity: Math.random() * 0.4 + 0.5, // Increased from 0.2-0.5 to 0.5-0.9
      };
    }

    // Animation loop
    function animate() {
      if (!canvas || !ctx || !isEnabled) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach((particle, index) => {
        // Update position
        particle.y += particle.speed;
        particle.rotation += particle.rotationSpeed;

        // Reset particle if it goes off screen
        if (particle.y > canvas.height + particle.size) {
          particlesRef.current[index] = createParticle();
        }

        // Draw particle
        ctx.save();
        ctx.translate(particle.x, particle.y);
        ctx.rotate(particle.rotation);
        ctx.globalAlpha = particle.opacity;

        // Get the appropriate image
        const image = particle.type === "divine" ? currencyImages.divine : currencyImages.mirror;
        
        if (image && image.complete && image.naturalWidth > 0) {
          // Draw the actual currency icon
          ctx.drawImage(
            image,
            -particle.size / 2,
            -particle.size / 2,
            particle.size,
            particle.size
          );
        }
        // If image not loaded, don't draw anything (removed fallback shapes)

        ctx.restore();
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    }

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isEnabled, currencyImages]);

  // Don't render on server to prevent hydration mismatch
  if (!isMounted || !isEnabled) return null;

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{ opacity: 0.9 }}
    />
  );
}

